import cv2


def delivery_report(err, msg):
    if err:
        print("Failed to deliver message: {0}: {1}"
              .format(msg.value(), err.str()))
    else:
        print(f"msg produced. \n"
                    f"Topic: {msg.topic()} \n" +
                    f"Partition: {msg.partition()} \n" +
                    f"Offset: {msg.offset()} \n" +
                    f"Timestamp: {msg.timestamp()} \n")
                    
def serializeImg(img):
    _, img_buffer_arr = cv2.imencode(".jpg", img)
    img_bytes = img_buffer_arr.tobytes()
    return img_bytes
    
def reset_map(_dict):
    for _key in _dict:
        _dict[_key] = []
    return _dict

def create_collections_unique(db, video_names):
    videos_map = {}
    for video in video_names:
        video_collection = db[video]
        video_collection.create_index("frame", unique=True)
        videos_map.update({video: []})
    return videos_map