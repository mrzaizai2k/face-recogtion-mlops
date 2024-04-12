"""Utility functions to support APP"""
import base64
import numpy as np



# G. 1.
def np_to_json(obj, prefix_name=""):
    """Serialize numpy.ndarray obj
    Should send base64 image with dtype and shape
    :param prefix_name: unique name for this array.
    :param obj: numpy.ndarray"""
    json_data = {
        f"{prefix_name}_frame": base64.b64encode(obj.tostring()).decode("utf-8"),
        f"{prefix_name}_dtype": obj.dtype.str,
        f"{prefix_name}_shape": obj.shape
        }
    return json_data

# G. 2.
def np_from_json(obj, prefix_name=""):
    """Deserialize numpy.ndarray obj
    :param prefix_name: unique name for this array.
    :param obj: numpy.ndarray"""
    np_array = np.frombuffer(base64.b64decode(obj[f"{prefix_name}_frame"].encode("utf-8")),
                         dtype=np.dtype(obj[f"{prefix_name}_dtype"])).reshape(obj[f"{prefix_name}_shape"])
    return np_array


