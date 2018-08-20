import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from keras import optimizers
import random
import numpy as np
import math

reward_dep = .7
x_train = True

model = Sequential()
model.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model_2 = Sequential()
model_2.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

def one_hot(state):
	current_state = []

	for square in state:
		if square == 0:
			current_state.append(1)
			current_state.append(0)
			current_state.append(0)
		elif square == 1:
			current_state.append(0)
			current_state.append(1)
			current_state.append(0)
		elif square == -1:
			current_state.append(0)
			current_state.append(0)
			current_state.append(1)

	return current_state

def get_outcome(state):
	total_reward = 0

	for i in range(0, 9):
		if i == 0 or i == 3 or i == 6:
			if state[i] == state[i + 1] and state[i] == state[i + 2]:
				total_reward = state[i]					
				break
			elif state[0] == state[4] and state[0] == state[8] and i == 0:
				total_reward = state[0]
				break
		if i < 3:
			if state[i] == state[i + 3] and state[i] == state[i + 6]:
				total_reward = state[i]					
				break
			elif state[2] == state[4] and state[2] == state[6] and i == 2:
				total_reward = state[2]
				break

	if (state[0] == state[1] == state[2]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[3] == state[4] == state[5]) and not state[3] == 0:
		total_reward = state[3]	
	elif (state[6] == state[7] == state[8]) and not state[6] == 0:
		total_reward = state[6]	
	elif (state[0] == state[3] == state[6]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[1] == state[4] == state[7]) and not state[1] == 0:
		total_reward = state[1]	
	elif (state[2] == state[5] == state[8]) and not state[2] == 0:
		total_reward = state[2]	
	elif (state[0] == state[4] == state[8]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[2] == state[4] == state[6]) and not state[2] == 0:
		total_reward = state[2]

	return total_reward

try:
	model = load_model('tic_tac_toe.h5')
	model_2 = load_model('tic_tac_toe_2.h5')
	print('Pre-existing model found... loading data.')
except:
	pass

def process_games(games, model, model_2):
	global x_train
	xt = 0
	ot = 0
	dt = 0
	states = []
	q_values = []
	states_2 = []
	q_values_2 = []

	for game in games:
		total_reward = get_outcome(game[len(game) - 1])
		if total_reward == -1:
			ot += 1
		elif total_reward == 1:
			xt += 1
		else:
			dt += 1
		# print('------------------')
		# print(game[len(game) - 1][0], game[len(game) - 1][1], game[len(game) - 1][2])
		# print(game[len(game) - 1][3], game[len(game) - 1][4], game[len(game) - 1][5])
		# print(game[len(game) - 1][6], game[len(game) - 1][7], game[len(game) - 1][8])
		# print('reward =', total_reward)

		for i in range(0, len(game) - 1):
			if i % 2 == 0:
				for j in range(0, 9):
					if not game[i][j] == game[i + 1][j]:
						reward_vector = np.zeros(9)
						reward_vector[j] = total_reward*(reward_dep**(math.floor((len(game) - i) / 2) - 1))
						# print(reward_vector)
						states.append(game[i].copy())
						q_values.append(reward_vector.copy())
			else:
				for j in range(0, 9):
					if not game[i][j] == game[i + 1][j]:
						reward_vector = np.zeros(9)
						reward_vector[j] = -1*total_reward*(reward_dep**(math.floor((len(game) - i) / 2) - 1))
						# print(reward_vector)
						states_2.append(game[i].copy())
						q_values_2.append(reward_vector.copy())

	if x_train:
		zipped = list(zip(states, q_values))
		random.shuffle(zipped)
		states, q_values = zip(*zipped)
		new_states = []
		for state in states:
			new_states.append(one_hot(state))

		# for i in range(0, len(states)):
			# print(new_states[i], states[i], q_values[i])
			# print(np.asarray(new_states))

		model.fit(np.asarray(new_states), np.asarray(q_values), epochs=4, batch_size=len(q_values), verbose=1)
		model.save('tic_tac_toe.h5')
		del model
		model = load_model('tic_tac_toe.h5')
		print(xt/20, ot/20, dt/20)
	else:
		zipped = list(zip(states_2, q_values_2))
		random.shuffle(zipped)
		states_2, q_values_2 = zip(*zipped)
		new_states = []
		for state in states_2:
			new_states.append(one_hot(state))

		# for i in range(0, len(states)):
			# print(new_states[i], states[i], q_values[i])
			# print(np.asarray(new_states))

		model_2.fit(np.asarray(new_states), np.asarray(q_values_2), epochs=4, batch_size=len(q_values_2), verbose=1)
		model_2.save('tic_tac_toe_2.h5')
		del model_2
		model_2 = load_model('tic_tac_toe_2.h5')
		print(xt/20, ot/20, dt/20)

	x_train = not x_train

# win = 1; draw = 0; loss = -1 --> moves not taken are 0 in q vector



mode = input('Choose a mode: (training/playing) ')

while True:
	board = [0, 0, 0, 0,  0, 0, 0, 0, 0]
	# sides --> 0 = Os, 1 = Xs
	games = []
	current_game = []

	if mode == 'training':
		print(x_train)
		# total_games = int(input('How many games should be played? '))
		total_games = 2000
		# e_greedy = float(input('What will the epsilon-greedy value be? '))
		e_greedy = .7

		for i in range(0, total_games):
			playing = True
			nn_turn = True
			c = 0
			board = [0, 0, 0, 0,  0, 0, 0, 0, 0]
			# sides --> 0 = Os, 1 = Xs
			current_game = []
			current_game.append(board.copy())
			nn_board = board

			while playing:
				if nn_turn:
					if random.uniform(0, 1) <= e_greedy:
						choosing = True
						while choosing:
							c = random.randint(0, 8)
							if board[c] == 0:
								choosing = False
								board[c] = 1
								current_game.append(board.copy())
								# save state to game array
					else:
						pre = model.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
						highest = -1000
						num = -1
						for j in range(0, 9):
							if board[j] == 0:
								if pre[j] > highest:
									highest = pre[j].copy()
									num = j

						choosing = False
						board[num] = 1
						current_game.append(board.copy())

				else:
					if random.uniform(0, 1) <= e_greedy:
						choosing = True
						while choosing:
							c = random.randint(0, 8)
							if board[c] == 0:
								choosing = False
								board[c] = -1
								current_game.append(board.copy())
								# save state to game array
					else:
						pre = model_2.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
						highest = -1000
						num = -1
						for j in range(0, 9):
							if board[j] == 0:
								if pre[j] > highest:
									highest = pre[j].copy()
									num = j

						choosing = False
						board[num] = -1
						current_game.append(board.copy())

				playable = False

				for square in board:
					if square == 0:
						playable = True
					# elif find square and check

				if not get_outcome(board) == 0:
					playable = False

				# print(get_outcome(board))

				if not playable:
					playing = False

				nn_turn = not nn_turn

				# print(board[0], board[1], board[2])
				# print(board[3], board[4], board[5])
				# print(board[6], board[7], board[8])

			games.append(current_game)
			# print('current game:', current_game)

		process_games(games, model, model_2)
	elif mode == 'playing':
		print('')
		print('A new game is starting!')
		print('')

		team = input('Choose a side: (x/o) ')
		print('')

		board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		running = True
		x_turn = True
		while running:
			if (x_turn and team == 'o') or (not x_turn and not team == 'o'):
				if team == 'o':
					pre = model.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
				elif team == 'x':
					pre = model_2.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
				# print(pre)
				print('')
				highest = -1000
				num = -1
				for j in range(0, 9):
					if board[j] == 0:
						if pre[j] > highest:
							highest = pre[j].copy()
							num = j

				print(pre)

				# TODO: ADD EXTRA IF STATEMENT FOR NUM == -1 (FIRST OPTION ALWAYS TRUMPS)

				if team == 'o':
					board[num] = 1
				elif team == 'x':
					board[num] = -1
				x_turn = not x_turn
				print('AI is thinking...')
			else:
				move = int(input('Input your move: '))
				if board[move] == 0:
					if team == 'o':
						board[move] = -1
					elif team == 'x':
						board[move] = 1
					x_turn = not x_turn
				else:
					print('Invalid move!')

			r_board = []

			for square in board:
				if square == 0:
					r_board.append('-')
				elif square == 1:
					r_board.append('x')
				elif square == -1:
					r_board.append('o')

			print(r_board[0], r_board[1], r_board[2])
			print(r_board[3], r_board[4], r_board[5])
			print(r_board[6], r_board[7], r_board[8])
			
			full = True

			for square in board:
				if square == 0:
					full = False

			if full:
				running = False
				if get_outcome(board) == 0:
					print('The game was drawn!')

			if not get_outcome(board) == 0:
				running = False
				print(get_outcome(board), 'won the game!')