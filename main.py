game = 0
cust1 = 'ABACAD' * 8

train_pattern = cust1
train_iterations = 100
train_seq = train_pattern * train_iterations
games = len(train_seq)

win_obj_A = (0, 0)
win_obj_B = (0, 4)
win_obj_C = (4, 0)
win_obj_D = (4, 4)


def set_win_pos(letter):
    if letter == "A":
        winning_pos = win_obj_A
    elif letter == "B":
        winning_pos = win_obj_B
    elif letter == "C":
        winning_pos = win_obj_C
    else:
        winning_pos = win_obj_D
    return winning_pos


def check_to_int():
    if int_move_counter < 4 and into_int_state:
        return True
    return False


while game < games:
    scenario = train_seq[game]
    win_pos = set_win_pos(scenario)
    go_to_int = check_to_int()
    if go_to_int:
        int_action = pick_int_move(prev_scenario)
        current_pos = take_next_move(int_action)
        int_move_counter += 1
    else:
        into_int_state = False
        if current_pos == win_pos:
            reward = base_reward * (get_exp(expval * act_move_counter))
            # todo: apply rewards
            game += 1
            # todo: reset environment
            prev_scenario = scenario
        else:
            act_action = pick_act_move(scenario)
            current_pos = take_next_move(act_action)
            act_move_counter += 1