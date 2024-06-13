from numba import cuda

# Print all available GPUs and their basic details
for i, gpu in enumerate(cuda.gpus):
    with gpu:
        print(f"GPU {i}:")
        device = cuda.get_current_device()
        print(f"  Name: {device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
        print(f"  Total Memory: {device.total_memory / 1e9:.2f} GB")
        print(f"  Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
        print(f"  Max Block Dimensions: {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}")
        print(f"  Max Grid Dimensions: {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}")
        print()