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
        
    def AcceptInput(self, T):
        tt_lst = [0] * T
        for i in range(T):
            tt_lst[i] = int(input())
            
        for val in tt_lst:
            self.CreatePattern(val)
        
        
sol = Solution()
T = int(input())
sol.AcceptInput(T)