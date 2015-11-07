# lda_gibbs
This is a lda implement using gibbs sampling.

lda.py包含中英文语料的处理，使用时在命令行输入以下内容（k为主题数，i为迭代次数，alpha和beta为超参数）；

$ ./lda.py -f filename -k 10 --alpha=0.5 --beta=0.5 -i 50

即可打印出语料的top主题词，以及输出在model文件夹中，final.phi，final.theta，final.twords分别存放主题词分布，文档主题分布以及top主题词
