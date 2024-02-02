import torch
import gc
import psutil

@torch.no_grad()
def free_storage_(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)
        
@torch.no_grad()
def alloc_storage_(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():  # no need to reallocate
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())

## PyTorch 2.0

# @torch.no_grad()
# def free_storage_(data: torch.Tensor) -> None:
#     """Free underlying storage of a Tensor."""
#     assert data.storage_offset() == 0, "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
#     data._typed_storage()._resize_(0)

# @torch.no_grad()
# def alloc_storage_(data:torch.Tensor, size: torch.Size) -> None:
#     tensor_storage_size = data._typed_storage()._size()
#     assert tensor_storage_size == 0, f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}"
#     data._typed_storage()._resize_(size.numel())

def see_memory_usage(rank = 0):

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(
        f"rank: {rank},  MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2)} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2)} GB \
        Peak_RM {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2)} GB")

    # vm_stats = psutil.virtual_memory()
    # used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    # print(
    #     f'rank: {rank}, CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # # get the peak memory to report correct data, so reset the counter for the next call
    # if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
    #     torch.cuda.reset_peak_memory_stats()