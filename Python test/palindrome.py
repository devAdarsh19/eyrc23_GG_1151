class Solution:
    #User fcn to check if palindrome
    def palindrome(self, inp_str):
        inp_str_lower = inp_str.lower()
        if (inp_str_lower == inp_str_lower[::-1]):
            print('\nIt is a palindrome')
        else:
            print('\nIt is not a palindrome')

    #User fcn to take input and call palindrome on this input
    def AcceptInput(self, n):
        inp_list = [0] * n
        for i in range(n):
            inp_list[i] = input('Enter string to check if palindrome : ')

        for i in range(n):
            self.palindrome(inp_list[i])

sol = Solution()
n = int(input('Number of strings to take : '))
sol.AcceptInput(n)