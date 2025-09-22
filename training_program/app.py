import torch
import time

def main():
    if not torch.cuda.is_available():
        print("GPU not available. Exiting.")
        return

    device = torch.device("cuda:0")

    iteration = 0
    while True:
        iteration += 1

        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)

        print(f"Iteration {iteration}: Result tensor shape: {c.shape}")
        time.sleep(2)

if __name__ == "__main__":
    main()
