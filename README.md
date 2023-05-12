# My-AutoPano

Run Instructions

Open <Folder Path>
in our case schavan_schatterjee_p1


Phase1:

Navigate to Phase1 folder
Navigate to Code Folder

python3 Wrapper.py —path Train/Set1 —save True
python3 Wrapper.py —path Test/TestSet1 —save True


Phase2:

Navigate to Phase2 folder
Navigate to Code Folder

To Train Supervised- 
python3 Train.py –ModelType ‘sup’

To Train Unsupervised-
python3 Train.py –ModelType ‘unsup’

To Test Supervised- 
python3 Test.py –ModelType ‘sup’

To Test Unsupervised-
python3 Test.py –ModelType ‘unsup’
