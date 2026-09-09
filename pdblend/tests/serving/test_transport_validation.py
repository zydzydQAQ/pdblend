from ecopadg.serving.transport_validation import check_events


def test_cross_tp_tensor_completeness_detects_corruption_and_duplicate_imports():
    events=[]
    for direction in ('export','import'):
        for index in range(9):
            events.append(dict(direction=direction,tensor_id='abc#'+str(index),target_rank=0,
                               sha256=str(index),shape=[48,128,2,128],dtype='torch.bfloat16'))
    events.append(dict(direction='cache_readback',nonce='abc',target_rank=0,bit_exact=True))
    assert check_events(events,'abc',4,1)['transport_bit_exact']
    assert check_events(events,'abc',4,1)['cache_bit_exact']
    duplicate=events+[dict(events[9])]
    assert not check_events(duplicate,'abc',4,1)['transport_bit_exact']
    events[9]['sha256']='corrupt'
    assert not check_events(events,'abc',4,1)['transport_bit_exact']
