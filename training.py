import pulumi_awsx as awsx


def training_image() -> str:
    ecr_repo = awsx.ecr.Repository("training-image",
        force_delete=True
    )
    
    image = awsx.ecr.Image("training-image",
        repository_url=ecr_repo.repository.repository_url,
        context="./training_program",
        dockerfile="./training_program/Dockerfile"
    )
    
    return image.image_uri
