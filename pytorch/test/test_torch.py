def can_cast(src_dtype, dst_dtype):
            # torch.can_cast(torch.int16, torch.uint8) returns True
            # which isn't actually safe-cast.
            # This function returns False in this case.
            def is_unsigned_int(dtype):
                return dtype is torch.uint8

            if is_unsigned_int(dst_dtype):
                return is_unsigned_int(src_dtype)
            return torch.can_cast(src_dtype, dst_dtype)

