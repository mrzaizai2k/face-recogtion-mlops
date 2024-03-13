# face-recogtion-mlops
make the whole system for face recognition with the scale of over 300 cameras

Follow this [link](https://github.com/Xuanhoang214/WSL2-USB-CAMERA) to connect usb camera to wsl 

to take pose, vào .venv/Lib/site-packages/compreface/common/typed_dict.py and change

        if row.find('age') == -1 and row.find('calculator') == -1 and row.find('gender') == -1 \
                    and row.find('landmarks') == -1 and row.find('mask') == -1 and row.find('pose') == -1: