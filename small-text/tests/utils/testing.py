def assert_list_of_tensors_equal(unittest_obj, input, other):
    import torch
    tensor_pairs = zip([item for item in input], [item for item in other])
    is_equal = [torch.equal(first, second)
                for first, second in tensor_pairs]
    unittest_obj.assertTrue(np.alltrue(is_equal))

