import csv

def corporate(label, number):
    c_f = open('train_c.tsv', 'a+', encoding='utf-8', newline='')
    writer = csv.writer(c_f, delimiter='\t')
    gc_f = open('../'+label+'_contextc.tsv', encoding='utf-8')
    reader = csv.reader(gc_f, delimiter='\t')

    for i, x in enumerate(reader):
        if i==number:
            break
        writer.writerow([label, x[1].replace("\n","").replace("\t","")])


corporate('advise', 50)
corporate('effect', 50)
corporate('mechanism', 50)
corporate('int', 100)