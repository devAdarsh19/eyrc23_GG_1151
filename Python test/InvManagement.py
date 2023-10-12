class Solution:
    
    def AddElements(self, item_info, item, qty):
        if item not in item_info.keys():
            item_info[item] = int(qty)
        else:
            item_info[item] += int(qty)
        print(f'ADDED Item {item}')
        return item_info
        # Description : If item does not already exist in inventory, add item and its qty to inventory. If it  
        # exists, update qty. Then return the dictionary item_info

    def DeleteElements(self, item_info, item, qty):
        if item not in item_info:
            print(f'{item} does not exist')
        else:
            if int(qty) > item_info[item]:
                print(f'Item {item} could not be DELETED')
            else:
                item_info[item] -= int(qty)
                print(f'DELETED Item {item}')
        return item_info
        # Description : If item does not exist in inventory, display '$item$ does not exist'. If it does and the  
        # qty to be deleted is greater what is available, print $item$ could not be deleted. And if qty to be 
        # deleted is less than existing qty, delete that qty of items from inventory

    def AcceptInput(self, T):
        for i in range(T):
            item_info = {}
            item_info_lst_nested = [[]]
            #Elements already present in inventory
            N = int(input())
            for entry in range(N):
                item, qty = input().split(' ')
                item_info[item] = int(qty)

            #Items to be added
            operation_lst = []
            M = int(input())
            for entry in range(M):
                operation_lst.append(input())

            for operation in operation_lst:
                #splitting into operation, item and quantity
                operation, item, qty = operation.split(' ')

                if operation == 'ADD': #if operation is ADD
                    item_info = self.AddElements(item_info, item, qty)
                elif operation == 'DELETE': #if operation is DELETE
                    item_info = self.DeleteElements(item_info, item, qty)

            # Finding the total number of items in inventory
            # total_items = 0
            # for qty in item_info.values():
            #     total_items += qty
            print('Total Items in Inventory:', sum(item_info.values()))

sol = Solution()
T = int(input())
sol.AcceptInput(T)