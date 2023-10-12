class Solution:
    def CreatePattern(self, N):
        ptrn_lst = ['*'] * N
        for i in range(4, N, 5):
            ptrn_lst[i] = '#';
        
        while (N != 0):
            for i in range(N+1):
                for j in range(N):
                    print(ptrn_lst[j], end='')
                print(sep='\n')
                N -= 1
            break

sol = Solution()
sol.CreatePattern(14)

