from typing import Optional, List, Union, Tuple

class BlockManager():
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(1, num_blocks))
        self.req_to_block_ids = {}
        # Track sequence info: (block_ids, used_slots_in_last_block, slot_ids)
        self.seq_info = {}
    
    def available_size(self):
        return len(self.free_blocks) * self.block_size

    def allocate(self, req_id, need_size: int) -> Optional[List[int]]:
        """Allocate slots for a sequence.
        
        Args:
            req_id: Request ID for the sequence
            need_size: Number of tokens needed for the sequence
            
        Returns:
            List of slot IDs, or None if allocation failed
        """
        remaining_tokens = need_size
        allocated_slots = []
        new_blocks = []

        # If sequence exists, try to fill remaining space in last block
        if req_id in self.seq_info:
            block_ids, used_slots, slot_ids = self.seq_info[req_id]
            last_block = block_ids[-1]
            available_in_last = self.block_size - used_slots
            
            tokens_to_add = min(remaining_tokens, available_in_last)
            if tokens_to_add > 0:
                start_slot = last_block * self.block_size + used_slots
                new_slots = list(range(start_slot, start_slot + tokens_to_add))
                allocated_slots.extend(new_slots)
                remaining_tokens -= tokens_to_add
                used_slots += tokens_to_add
                self.seq_info[req_id] = (block_ids, used_slots, slot_ids + new_slots)

        # Need new blocks
        if remaining_tokens > 0:
            blocks_needed = (remaining_tokens + self.block_size - 1) // self.block_size
            
            if blocks_needed > len(self.free_blocks):
                return None
                
            new_blocks = self.free_blocks[:blocks_needed]
            self.free_blocks = self.free_blocks[blocks_needed:]

            # Calculate slots in new blocks
            for i, block_id in enumerate(new_blocks):
                tokens_in_block = min(self.block_size, remaining_tokens)
                start_slot = block_id * self.block_size
                new_slots = list(range(start_slot, start_slot + tokens_in_block))
                allocated_slots.extend(new_slots)
                remaining_tokens -= tokens_in_block

                # Update sequence info
                if req_id in self.seq_info:
                    existing_blocks, _, existing_slot_ids = self.seq_info[req_id]
                    all_blocks = existing_blocks + [block_id]
                    all_slot_ids = existing_slot_ids + new_slots
                else:
                    all_blocks = [block_id]
                    all_slot_ids = new_slots
                self.seq_info[req_id] = (all_blocks, tokens_in_block, all_slot_ids)

            # Update block mapping
            if req_id in self.req_to_block_ids:
                self.req_to_block_ids[req_id].extend(new_blocks)
            else:
                self.req_to_block_ids[req_id] = new_blocks

        return allocated_slots, new_blocks

    def free(self, req_id):
        """Free all blocks allocated to a request.
        
        Args:
            req_id: Request ID whose blocks should be freed
        """
        if req_id in self.req_to_block_ids:
            freed_blocks = self.req_to_block_ids[req_id]
            self.free_blocks.extend(freed_blocks)
            del self.req_to_block_ids[req_id]
            del self.seq_info[req_id]


# from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

class HPUTokenToKVPool():
    def __init__(self, vllm_cache_engine, block_manager: BlockManager):
        self.vllm_cache_engine = vllm_cache_engine
        self.block_manager = block_manager
    
    def alloc_prefill(self, id_len_pairs: List[Tuple[int, int]]):
        slot_ids = []
        for seq_id, need_size in id_len_pairs:
            slots, new_blocks = self.block_manager.allocate(seq_id, need_size)
            print(f"seq_id: {seq_id}, need_size: {need_size}, slots: {slots}")
            print(f"block_manager.seq_info: {self.block_manager.seq_info}")
            slot_ids.extend(slots)
        return slot_ids

    def alloc_decode(self, seq_ids: List[int]):
        slot_ids = []
        for seq_id in seq_ids:
            slots, new_blocks = self.block_manager.allocate(seq_id, 1)
            print(f"seq_id: {seq_id}, slots: {slots}")
            print(f"block_manager.seq_info: {self.block_manager.seq_info}")
            slot_ids.extend(slots)
        return slot_ids
    
    def available_size(self):
        return self.block_manager.available_size()

class HPUReqToTokenPool():
    
    def __init__(self, block_manager: BlockManager, size):
        self.block_manager = block_manager
        self.size = size
        self.block_size = self.block_manager.block_size
        self.free_slots = list(range(size))
    
    # def write(self, indices, values):
    #     self.req_to_token[indices] = values

    def available_size(self):
        return self.size - len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))

    def get_token_indices(self, req_id):
        # return slot ids
        seq_info = self.block_manager.seq_info[req_id]
        _, _, slot_ids = seq_info
        return slot_ids

# Test cases

def test_block_manager():
    """Test BlockManager functionality"""
    # Initialize with block size 4 and 3 blocks
    bm = BlockManager(block_size=4, num_blocks=4)
    
    # Test initial allocation
    slots = bm.allocate(req_id=1, need_size=3)
    assert slots == [0, 1, 2], f"Expected [0, 1, 2], got {slots}"
    assert bm.seq_info[1] == ([0], 3, [0, 1, 2]), f"Incorrect sequence info, got {bm.seq_info[1]}"
    assert bm.free_blocks == [1, 2, 3], f"Incorrect free slots, got {bm.free_blocks}"
    
    # Test allocation to same sequence using remaining space
    new_slots = bm.allocate(req_id=1, need_size=2)
    assert len(new_slots) == 2, f"Expected length 2, got {len(new_slots)}"
    assert new_slots == [3, 4], f"Expected [3, 4], got {new_slots}"
    assert bm.seq_info[1] == ([0, 1], 1, [0, 1, 2, 3, 4]), f"Incorrect sequence info, got {bm.seq_info[1]}"
    assert bm.free_blocks == [2, 3], f"Incorrect free slots, got {bm.free_blocks}"
    # Test allocation requiring new block
    slots = bm.allocate(req_id=2, need_size=5)
    assert len(slots) == 5, f"Expected 5 slots, got {len(slots)}"
    assert bm.seq_info[2] == ([2, 3], 1, [8, 9, 10, 11, 12]), f"Incorrect sequence info, got {bm.seq_info[2]}"
    assert bm.free_blocks == [], f"Incorrect free slots, got {bm.free_blocks}"
    # Test freeing
    bm.free(req_id=1)
    assert 0 in bm.free_blocks, "Block not returned to free slots"
    assert 1 not in bm.req_to_block_ids, "Request ID not removed"
    assert 1 not in bm.seq_info, "Sequence info not removed"
    
    assert bm.free_blocks == [0, 1], f"Incorrect free slots, got {bm.free_blocks}"
    # Test allocation after freeing
    slots = bm.allocate(req_id=3, need_size=2)
    assert slots == [0, 1], f"Expected [0, 1], got {slots}"
    assert bm.seq_info[3] == ([0], 2, [0, 1]), f"Incorrect sequence info, got {bm.seq_info[3]}"
    assert bm.free_blocks == [1], f"Incorrect free slots, got {bm.free_blocks}"

def run_all_tests():
    """Run all test cases"""
    test_block_manager()
    # test_hpu_token_to_kv_pool()
    # test_hpu_req_to_token_pool()
    # print("All tests passed!")

if __name__ == "__main__":
    run_all_tests()