from chk import interpret,b_func
import glob
path = './val/**.jpg'
cnt = 0
for pathload in glob.glob(path):
    A = interpret(pathload)
    B = int(pathload.lstrip("""./val\\t""").split('_')[0])

    if A!=B:
        print(pathload, A, B)
    else:
        cnt+=1

print(cnt/len(glob.glob(path)))

"""
5  -> 44/48
10 -> 43/48
15 -> 46/48
20 -> 44/48
25 -> 45/48
30 -> 46/48 
35 -> 44/48
인두기 이미지 수정
"""