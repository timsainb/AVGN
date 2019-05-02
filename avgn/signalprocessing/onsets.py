import numpy as np


def detect_onsets_offsets(data, threshold, min_distance):
    """
    detects when a when a signal jumps above zero, and when it goes back to zero
    """
    on = data > threshold  # when the data is greater than thresh
    left_on = np.concatenate(([0], on), axis=0)[0:-1]
    onset = np.squeeze(np.where(on & (left_on != True)))
    offset = np.squeeze(np.where((on != True) & (left_on == True)))

    # make sure there is an offset at some point...
    if data[-1] > threshold:
        offset = np.append(
            offset, len(data)
        )  

    if len(np.shape(onset)) < 1:
        offset = [offset]
        onset = [onset]

    new_offset = []
    new_onset = []
    if len(onset) < 1:
        offset = []
        onset = []
    else:
        new_onset.append(onset[0])

        if len(onset) > 1:
            for i in range(len(onset) - 1):
                if (onset[i + 1] - offset[i]) > min_distance:
                    new_onset.append(onset[i + 1])
                    new_offset.append(offset[i])

        new_offset.append(offset[-1])
    return np.atleast_1d(np.squeeze(new_onset)), np.atleast_1d(np.squeeze(new_offset))
