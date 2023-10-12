class Solution:
    def FindTopper(self, students):
        toppers = []
        for i in range(len(students)):
            students[i].split()
            students = map(int, students[i][2])
        
        for j in range(len(students)):
            pass #Start here

        

    def AcceptInput(self, testcases, details_count):
        student_details = [0] * details_count
        for i in range(testcases):
            for j in range(details_count):
                student_details[j] = input('Enter name and score with space : ')
