def add(item, quantity):
    
    if item in inventory:
    
        inventory[item] = inventory[item]+quantity
        
        print(f'UPDATED Item {item}')
        
    else:
        
        inventory[item] = quantity
        print(f'ADDED Item {item}')


def delete(item, quantity):
    if item in inventory:
        
        if inventory[item] >= quantity:
            
            inventory[item] -= quantity
            
            print(f'DELETED Item {item}')
            
        else:
            print(f'Item {item} could not be DELETED')
            
    else:
        print(f'Item {item} does not exist')
        


inp = int(input())

for i in range(inp):
    
    
    inventory = {}
    
    
   
    n = int(input())
    
  
    for _ in range(n):
        
        item, item_quantity = input().split()
        inventory[item] = int(item_quantity)
    
    
    m = int(input())
    
    
    for j in range(m):
        
        operation, item, quantity = input().split()
        
        quantity = int(quantity)
        
        
        if operation == 'ADD':
            add(item, quantity)
            
            
            
            
        elif operation == 'DELETE':
            delete(item, quantity)
            
    
    
    total = sum(inventory.values())
    
    # Print total quantity
    print(f'Total Items in Inventory: {total}')