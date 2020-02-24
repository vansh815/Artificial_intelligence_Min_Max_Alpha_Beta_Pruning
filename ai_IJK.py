#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: [vansh shah, Aditya Kartikeya, Prashanth Sateesh] [ID : vanshah, psateesh, admall]

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK


# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game 
#
# This function should analyze the current state of the game and determine the 
# best move for the current player. It should then call "yield" on that move.

def next_move(game: Game_IJK)-> None:

    '''board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    '''

    board = game.getGame()
    
    MAX = float('inf')
    MIN = float('-inf')
    
    def calculate_empty_tiles(state):
        bonus  = 0 
        for i in range(6):
            
            for j in range(6):
                if state[i][j] == 0 :
                    bonus = bonus + 1
        return bonus 
    
    def weighted_tiles(state):
        board_score = [6 * 35, 6 * 34, 6 * 33, 6 * 32, 6 * 31, 6 * 30,
                      6 * 24, 6 * 25, 6 * 26, 6 * 27, 6 * 28, 6 * 29,
                      6 * 23, 6 * 22, 6 * 21, 6 * 20, 6 * 19, 6 * 18,
                      6 * 12, 6 * 13, 6 * 14, 6 * 15, 6 * 16, 6 * 17,
                      6 * 11, 6 * 10, 6 * 9, 6 * 8, 6 * 7, 6 * 6,
                      6 * 0, 6 * 1, 6 * 2, 6 * 3, 6 * 4, 6 * 5]

        bonus = 0
        for i in range(36):
            current_element = ord(state[i])
            bonus = bonus +  int(current_element) * int(board_score[i])

        return bonus
    
    def convert_to_number(state):
        for i in range(0,6):
            for j in range(0,6) : 
                if state[i][j] == "A" or state[i][j] == "a":
                    state[i][j] = 1
                elif state[i][j] == "B" or state[i][j] == "b":
                    state[i][j] = 2
                elif state[i][j] == "C" or state[i][j] == "c":
                    state[i][j] = 3
                elif state[i][j] == "D" or state[i][j] == "d":
                    state[i][j] = 4
                elif state[i][j] == "E" or state[i][j] == "e":
                    state[i][j] = 5
                elif state[i][j] == "F" or state[i][j] == "f":
                    state[i][j] = 6
                elif state[i][j] == "G" or state[i][j] == "g":
                    state[i][j] = 7
                elif state[i][j] == "H" or state[i][j] == "h":
                    state[i][j] = 8
                elif state[i][j] == "I" or state[i][j] == "i":
                    state[i][j] = 9
                elif state[i][j] == "J" or state[i][j] == "j":
                    state[i][j] = 10
                elif state[i][j] == "K" or state[i][j] == "k":
                    state[i][j] = 11     
                else :
                    state[i][j] = 0 
        return state
    
    def calculate_monotonicity(state):
        bonus = 0
        for i in range(6):
            row = state[i]
            for j in range(6):
                x = row[j]
                # now check left and right side of the element 
                # call function to check the position of first non-zero number in right direction 
                if state[i][j] != 0 and j != 5 :   
                    for z in range(j+1 ,6):
                        if state[i][z] != 0 :  
                            index = z
                            break
                        elif z == 5 and state[i][z] == 0:
                            index = z
                    
                            
                    
                    
                
                 
                    if x <= row[index] and row[index] != 0  : 
                        bonus = bonus + 10
                    elif x > row[index] and row[index] != 0 :
                        bonus = bonus - 15
                
                
                    if x <= state[index][j] :
                        bonus = bonus + 10
                    elif x > state[index][j] : 
                        bonus = bonus - 15
                elif state[i][j] == 0 : 
                    bonus = bonus + 0
            
                if state[i][j] != 0 and i != 5 :   
                
                    for z in range(i + 1, 6) : 
                        if state[z][j] != 0:
                            index = z
                            break
                        elif z == 5 and state[z][j] == 0:
                            index = z
                    index = i       
                    if x <= state[index][j] and state[index][j] != 0  : 
                        bonus = bonus + 10
                    elif x > state[index][j] and state[index][j] != 0 :
                        bonus = bonus - 15
                
                
                elif state[i][j] == 0 : 
                    bonus = bonus + 0
                    
        return bonus
    
    def flattened(listt):
        flattened_list = []
        for j in listt:
            for i in j:
                flattened_list.append(i)
        
        return flattened_list

            
    def calculate_smoothness(state) : 
        bonus = 0 
        for i in range(6):
            row = state[i]
            for j in range(6):
                x = row[j]
                if j != 5 and (x - state[i][j + 1]) <= 1 : 
                    bonus = bonus + 5
                if i != 5 and (x - state[i + 1][j]) <= 1 : 
                    bonus = bonus + 5
                    
        return bonus 
    
    def gt_mov(array):
        for i in range(len(array)):
            array[i] = list(array[i])
            if i >= 0 and i < 64:
                array[i][1] = 'U'
            elif i >= 64 and i < 128:
                array[i][1] = 'D'
            elif i >= 128 and i < 192:
                array[i][1] = 'L'
            elif i >= 192 and i < 256:
                array[i][1] = 'R'

        return array
    
    def heuristic_containing_evrything(now , state) :
        score = 0
        state = convert_to_number(state)
        score1 = calculate_empty_tiles(state) * 100
        #score2 = weighted_tiles(now)
        score3 = calculate_monotonicity(state) * 100
        score4 = calculate_smoothness(state) * 200
        score = score1  + score3 + score4
        return score
    
    def successors(game1):
        moves = ['U', 'D', 'L', 'R']

        successor_s = []
        
        for i in moves:
            successor1 = game.makeMove(i)
            successor1 = successor1.getGame()
            successor_s.append((successor1,i))
        return successor_s

    
    

    def parse_full(game1):
        for_first_depth = successors(game1)
        for_second_depth = []
        
        for i in for_first_depth:
        
            for_second_depth.append(successors(i[0]))
        
        for_second_depth = flattened(for_second_depth)
        
        for_third_depth = []
        
        for i in for_second_depth:
        
            for_third_depth.append(successors(i[0]))
        
        for_third_depth = flattened(for_third_depth)
        for_fourth_depth = []
        
        for i in for_third_depth:
        
            for_fourth_depth.append(successors(i[0]))
        
        for_fourth_depth = flattened(for_fourth_depth)
        
        
        
        future_successors = []
        possible_moves = []
        
        for i in for_fourth_depth:
            future_successors.append(i[0])
            possible_moves.append(i[1])
        scores = calculate_score(future_successors)
        final = []
        for i in range(len(scores)):
            final.append((scores[i],possible_moves[i]))
        final = gt_mov(final)
        return final

    def calculate_score(arrStates):
        scores = []
        for i in arrStates:
            
            x = flattened(i)
            score = heuristic_containing_evrything(x, i)
            scores.append(score)
        return scores

    def mini_max_algo(depth, Index_of_node, maximizing_player, values, alpha, beta):
        if depth == 4:
            
            return values[Index_of_node][0], values[Index_of_node][1]
        if maximizing_player:
            best = MIN
            final_move = " "
            for i in range(0, 4):
                value, moves = mini_max_algo(depth + 1, Index_of_node * 4 + i, False, values, alpha, beta)
                
                if value > best:
                    best = value
                    final_move = moves
                alpha = max(alpha, best)

                if beta <= alpha:
                    break
            return (best, final_move)
        
        
        
        else:
        
            
            best = MAX
            final_move = " "
            for i in range(0, 4):
                
                value, moves = mini_max_algo(depth + 1, Index_of_node * 4 + i, True, values, alpha, beta)
                
                
                if value < best:
                
                    best = value
                    
                    
                    final_move = moves
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return (best, final_move)

    final_s = parse_full(board)
    score, move = mini_max_algo(0, 0, True, final_s, MIN, MAX)

    yield move
       
    

    


