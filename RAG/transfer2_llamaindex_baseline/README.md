This is a baseline of llamaindex.

# make a text file per topic from wikipedia for RAG
- Just copy dataset directory from FiDbaseline (we don't write any files into this directory, so the symbolic link is enough) ``ln -s {AIO3_FiD_baseline_dir}/dataset .``
- ``python prep_data.py``

# run llamaindex
- ``pip install -r requirements.txt`` to install required packages
- ``python run.py``
- Check the accuracy by ``python check_answer.py datasets/aio_02_dev_v1.0.jsonl result.json``
