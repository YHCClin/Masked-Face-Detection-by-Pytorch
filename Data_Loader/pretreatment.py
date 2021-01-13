
# 根据数据集生成txt文件
if __name__ == "__main__":
    yes = 1
    no = 0
    with open("../agedb_30_masked/lables_train.txt", 'w') as f:
        for num in range(12000):
            f.write(str(num) + ".jpg" + " " + str(yes) + "\n")
            yes, no = no, yes