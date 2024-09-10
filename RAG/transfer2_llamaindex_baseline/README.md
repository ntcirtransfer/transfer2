This is a baseline of llamaindex.

run ``pip install -r requirements.txt`` to install required packages
- First, just copy dataset directory from FiDbaseline (we don't write any files into this directory, so the symbolic link can be used)
- Second, run ``python prep_data.py``
- Third, run ``python run.py``
- Finally, check the accuracy by ``python check_answer.py datasets/aio_02_dev_v1.0.jsonl result.json``
