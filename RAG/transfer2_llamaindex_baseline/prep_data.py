import os
import gzip

def prep_data():
    print('step 1/3 prepare data, each file is a description about the same theme')

    os.makedirs('data',exist_ok=True)

    assert os.path.exists('datasets/wiki/jawiki-20220404-c400-large.tsv.gz'), 'copy or symbolic link of dataset dir from FiD baseline'

    with gzip.open('datasets/wiki/jawiki-20220404-c400-large.tsv.gz', mode='rt',encoding='utf-8') as f:
        theme = ''
        fout = None
        i = 0
        j = 0
        for line in f:
            i += 1
            if i == 1:
                continue

            line = line.replace('"','')
            d = line.rstrip().split('\t')
            if theme != d[2]:
                j += 1
                if fout is not None:
                    fout.close()
                fout = open('data/p{0:06d}.txt'.format(j),'wt')
                fout.write(d[2]+'\n')
                theme = d[2]
            fout.write(d[1]+'\n\n')


prep_data()
