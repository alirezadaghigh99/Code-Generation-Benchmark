def from_str(classification_type_str):
        if classification_type_str == 'single-label':
            return ClassificationType.SINGLE_LABEL
        elif classification_type_str == 'multi-label':
            return ClassificationType.MULTI_LABEL
        else:
            raise ValueError('Cannot convert string to classification type enum: '
                             f'{classification_type_str}')

