"""BC supervision selected by the sampled buffer, not the record's source."""

TRAINING_OBJECTIVE = "three_buffer_demo_bc_v1"


def bc_eligible(buffer_id):
    
    
    return buffer_id == 0
