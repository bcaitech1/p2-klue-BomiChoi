import pandas as pd
from collections import Counter

folder = "/opt/ml/code/ensemble"
ans1 = pd.read_csv(folder+'/submission12.csv')
ans2 = pd.read_csv(folder+'/submission16.csv')
ans3 = pd.read_csv(folder+'/submission9.csv')
ans4 = pd.read_csv(folder+'/submission17.csv')
ans5 = pd.read_csv(folder+'/submission13.csv')

pred = []
for i in range(1000):
    li = []
    li.append(ans1.iloc[i].item())
    li.append(ans2.iloc[i].item())
    li.append(ans3.iloc[i].item())
    li.append(ans4.iloc[i].item())
    li.append(ans5.iloc[i].item())
    
    c = Counter(li)
    mode = c.most_common(1)
    ans = mode[0][0]
    pred.append(ans)
    # print(li, ans)

output = pd.DataFrame(pred, columns=['pred'])
output.to_csv(folder+'/submission.csv', index=False)


    