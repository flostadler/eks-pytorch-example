import json
import pulumi
import pulumi_awsx as awsx
import pulumi_eks as eks
import pulumi_kubernetes as k8s
import pulumi_aws as aws

import training


training_image_uri = training.training_image()


# Managed policy ARNs for EKS worker nodes
managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
]

def create_role(name: str) -> aws.iam.Role:
    """Creates a role and attaches the EKS worker node IAM managed policies"""
    role = aws.iam.Role(name, 
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"}
            }]
        })
    )
    
    for counter, policy in enumerate(managed_policy_arns):
        aws.iam.RolePolicyAttachment(f"{name}-policy-{counter}",
            policy_arn=policy,
            role=role
        )
    
    # Create custom ECR policy for the node role
    ecr_policy = aws.iam.Policy(f"{name}-ecr-policy",
        description="Custom ECR policy for EKS worker nodes",
        policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:GetAuthorizationToken"
                    ],
                    "Resource": "*"
                }
            ]
        })
    )
    
    # Attach the custom ECR policy to the role
    aws.iam.RolePolicyAttachment(f"{name}-ecr-policy-attachment",
        policy_arn=ecr_policy.arn,
        role=role
    )
    
    return role

# Get some values from the Pulumi configuration (or use defaults)
config = pulumi.Config()
min_cluster_size = config.get_int("minClusterSize", 3)
max_cluster_size = config.get_int("maxClusterSize", 6)
desired_cluster_size = config.get_int("desiredClusterSize", 3)
eks_node_instance_type = config.get("eksNodeInstanceType", "t3.medium")
vpc_network_cidr = config.get("vpcNetworkCidr", "10.0.0.0/16")

node_role = create_role("node-role")

# Create a VPC for the EKS cluster
eks_vpc = awsx.ec2.Vpc("eks-vpc",
    enable_dns_hostnames=True,
    cidr_block=vpc_network_cidr)

# Create the EKS cluster
eks_cluster = eks.Cluster("eks-cluster",
    vpc_id=eks_vpc.vpc_id,
    authentication_mode=eks.AuthenticationMode.API,
    public_subnet_ids=eks_vpc.public_subnet_ids,
    private_subnet_ids=eks_vpc.private_subnet_ids,
    instance_type=eks_node_instance_type,
    desired_capacity=desired_cluster_size,
    min_size=min_cluster_size,
    max_size=max_cluster_size,
    node_associate_public_ip_address=False,
    endpoint_private_access=False,
    endpoint_public_access=True
    )

# Create a managed node group for GPU instances
managed_node_group_gpu = eks.ManagedNodeGroup("al-2023-mng-nvidia-gpu",
    cluster=eks_cluster,
    node_role_arn=node_role.arn,
    subnet_ids=eks_vpc.private_subnet_ids,
    operating_system=eks.OperatingSystem.RECOMMENDED,
    gpu=True,
    instance_types=["g4dn.xlarge"],
    scaling_config=aws.eks.NodeGroupScalingConfigArgs(
        desired_size=1,
        max_size=3,
        min_size=1
    ),
    disk_size=500,
    labels={
        "nvidia-device-plugin-enabled": "true"
    },
    tags={
        "k8s.io/cluster-autoscaler/enabled": "true",
        "k8s.io/cluster-autoscaler/eks-cluster": eks_cluster.eks_cluster.name
    },
    taints=[aws.eks.NodeGroupTaintArgs(
        key="nvidia.com/gpu",
        value="true",
        effect="NO_SCHEDULE"
    )]
)

# Create a DaemonSet for the NVIDIA device plugin
nvidia_device_plugin = k8s.apps.v1.DaemonSet("nvidia-device-plugin",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="nvidia-device-plugin-daemonset",
        namespace="kube-system"
    ),
    spec=k8s.apps.v1.DaemonSetSpecArgs(
        selector=k8s.meta.v1.LabelSelectorArgs(
            match_labels={
                "name": "nvidia-device-plugin-ds"
            }
        ),
        update_strategy=k8s.apps.v1.DaemonSetUpdateStrategyArgs(
            type="RollingUpdate"
        ),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                labels={
                    "name": "nvidia-device-plugin-ds"
                }
            ),
            spec=k8s.core.v1.PodSpecArgs(
                tolerations=[k8s.core.v1.TolerationArgs(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule"
                )],
                node_selector={
                    "nvidia-device-plugin-enabled": "true"
                },
                priority_class_name="system-node-critical",
                containers=[k8s.core.v1.ContainerArgs(
                    name="nvidia-device-plugin-ctr",
                    image="nvcr.io/nvidia/k8s-device-plugin:v0.17.0",
                    env=[k8s.core.v1.EnvVarArgs(
                        name="FAIL_ON_INIT_ERROR",
                        value="false"
                    )],
                    security_context=k8s.core.v1.SecurityContextArgs(
                        allow_privilege_escalation=False,
                        capabilities=k8s.core.v1.CapabilitiesArgs(
                            drop=["ALL"]
                        )
                    ),
                    volume_mounts=[k8s.core.v1.VolumeMountArgs(
                        name="device-plugin",
                        mount_path="/var/lib/kubelet/device-plugins"
                    )]
                )],
                volumes=[k8s.core.v1.VolumeArgs(
                    name="device-plugin",
                    host_path=k8s.core.v1.HostPathVolumeSourceArgs(
                        path="/var/lib/kubelet/device-plugins"
                    )
                )]
            )
        )
    ),
    opts=pulumi.ResourceOptions(provider=eks_cluster.core.provider)
)

# Create PyTorch Deployment
pytorch_deployment = k8s.apps.v1.Deployment("pytorch",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="pytorch"
    ),
    spec=k8s.apps.v1.DeploymentSpecArgs(
        replicas=1,
        selector=k8s.meta.v1.LabelSelectorArgs(
            match_labels={
                "app": "pytorch"
            }
        ),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                labels={
                    "app": "pytorch"
                }
            ),
            spec=k8s.core.v1.PodSpecArgs(
                tolerations=[k8s.core.v1.TolerationArgs(
                    key="nvidia.com/gpu",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule"
                )],
                containers=[k8s.core.v1.ContainerArgs(
                    name="pytorch",
                    image=training_image_uri,
                    image_pull_policy="Always",
                    resources=k8s.core.v1.ResourceRequirementsArgs(
                        requests={
                            "cpu": "2",
                            "memory": "2Gi",
                            "nvidia.com/gpu": "1"
                        },
                        limits={
                            "cpu": "4",
                            "memory": "4Gi",
                            "nvidia.com/gpu": "1"
                        }
                    )
                )]
            )
        )
    ),
    opts=pulumi.ResourceOptions(provider=eks_cluster.core.provider)
)

# Export values to use elsewhere
pulumi.export("kubeconfig", eks_cluster.kubeconfig)
pulumi.export("vpcId", eks_vpc.vpc_id)
