import csv

def corporate(label, number):
    print('bgein:{}'.format(label))
    c_f_1 = open('test_advise.tsv', 'w', encoding='utf-8', newline='')
    c_f_2 = open('test_effect.tsv', 'w', encoding='utf-8', newline='')
    c_f_3 = open('test_mechanism.tsv', 'w', encoding='utf-8', newline='')
    c_f_4 = open('test_int.tsv', 'w', encoding='utf-8', newline='')
    writer_1 = csv.writer(c_f_1, delimiter='\t')
    writer_2 = csv.writer(c_f_2, delimiter='\t')
    writer_3 = csv.writer(c_f_3, delimiter='\t')
    writer_4 = csv.writer(c_f_4, delimiter='\t')
    gc_f = open('test_c.tsv', encoding='utf-8')
    reader = csv.reader(gc_f, delimiter='\t')


    for i, x in enumerate(reader):

        if x[0]=="advise":
            writer_1.writerow(["advise", x[1]])

        if x[0] == "effect":
            writer_2.writerow(["effect", x[1]])

        if x[0]=="mechanism":
            writer_3.writerow(["mechanism", x[1]])

        if x[0]=="int":
            writer_4.writerow(["int", x[1]])




corporate('advise', 600)
corporate('effect', 780)
corporate('mechanism', 1000)
corporate('int', 80)

