from math import log2
def KL(p,q):
	res = 0
	for i in range(0, len(p)):
		res += p[i]*log2(p[i]/q[i])
	return res
ABC = [[1/2,1/2], [1/4,3/4], [1/8,7/8]]
A, B, C = ABC[0], ABC[1], ABC[2]

# verify (a) and (c)
for i in range(0,3):
  for j in range(0,3):
    print(KL(ABC[i], ABC[j]))

# reject (b) and (d)
print(KL(A,B),KL(B,A))
print(KL(A,B)+KL(B,C),KL(A,C))
