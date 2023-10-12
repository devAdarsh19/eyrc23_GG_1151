class Solution:
    def CountLetters(self, line):
        new_line = line.replace('@', '')
        final_lst = []
        words = new_line.split(' ')
        for val in words:
            final_lst.append(str(len(val)))

        print(','.join(final_lst))
        

    def AcceptInput(self, T):
        str_lst = []
        for i in range(T):
            str_lst.append(input())
        for val in str_lst:
            self.CountLetters(val)
            print(sep='\n')
            

sol = Solution()

T = int(input())
sol.AcceptInput(T)