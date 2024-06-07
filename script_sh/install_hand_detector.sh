cd third-party/
git clone --depth 1 git@github.com:ddshan/hand_object_detector.git
cd hand_object_detector/
pip install -r requirements.txt
cd lib/
python setup.py build develop
cd ../../..