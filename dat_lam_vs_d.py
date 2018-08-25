import os

command_ori = r'F:\liziyao\Anaconda3\python sectne ' \
          r'--input .\data\flickr\links.txt '
output_ori = r'.\data\flickr\sectne_eta=1_ksize=1000_'


for lam in (0, 0.5, 1, 10, 20, 30, 50, 70, 100, 150, 200):
    for d in (64, 128, 256):
        output = output_ori + 'lam=%.1f_' % lam + \
            'dim=%d' % d + '.txt'
        command = command_ori + '--lam %.1f ' % lam + \
            '--dim %d ' % d + '--output %s' % output
        print(command)
        os.system(command)

