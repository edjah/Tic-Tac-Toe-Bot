import numpy as np
from termcolor import colored

# p1 = 1   p2 = -1
hidden_layer_mult = 3
class GameBoard:
	current_turn = 1
	game_num = 0
	p1_win_count = 0
	p2_win_count = 0
	draw_count = 0

	def __init__(self, size=3, ndim=2, human_mode=True, controller=None, show_computer_progress=True, training_mode=False):
		if size > 2:
			self.size = size
		else:
			self.size = 3
		if ndim > 1:
			self.ndim = ndim
		else:
			self.ndim = 2
		self.num_blocks = self.size ** self.ndim
		self.filled_spaces = 0
		self.found_winner = 0
		self.human_mode = human_mode
		self.controller = controller
		self.show_computer_progress = show_computer_progress
		self.training_mode = training_mode

		self.start_searching_for_winner = 2 * self.size - 1
		matrix_constructor = tuple([self.size for i in range(self.ndim)])
		self.board = np.zeros(matrix_constructor, dtype=int)
		self.plausible_moves = np.arange(self.board.size)

	def __str__(self):
		return self.board.__str__()

	def search_for_winner(self):
		self.found_winner = 0;
		for axis in range(self.ndim):
			sums = np.sum(self.board, axis=axis)
			if (max(sums) == self.size):
				self.found_winner = 1;
				break;
			elif (min(sums) == -self.size):
				self.found_winner = 2;
				break;
		if self.found_winner == 0:
			sum_diag1 = sum(self.board.diagonal())
			sum_diag2 = sum(np.fliplr(self.board).diagonal());
			if sum_diag1 == self.size:
				self.found_winner = 1
			elif sum_diag1 == -self.size:
				self.found_winner = 2
			if sum_diag2 == self.size:
				self.found_winner = 2
			elif sum_diag2 == -self.size:
				self.found_winner = 2

	def fill_space(self, num, pos):
		is_not_out_of_bounds = True
		for i in pos:
			if i >= self.size:
				is_not_out_of_bounds = False
				break;
		if is_not_out_of_bounds:	
			if self.board[pos] == 0:
				self.board[pos] = num
				number_to_remove = np.ravel_multi_index(pos, self.board.shape)
				index_to_remove = np.argwhere(self.plausible_moves == number_to_remove)[0][0]
				self.plausible_moves = np.delete(self.plausible_moves, index_to_remove)
				self.filled_spaces += 1
			else:
				pos = input('This space is filled. Try again: ')
				self.fill_space(num, pos)
		else:
			pos = input('This space is out of bounds. Try again: ')
			self.fill_space(num, pos)

	def requestInput(self, random=True):
		if random:
			return self.controller.provide_random_position()
		else:
			return self.controller.provide_educated_position(self.current_turn)

	def main(self):
		if self.human_mode:
			print(self)
		while (self.found_winner == 0) and (self.filled_spaces < self.num_blocks):
			if self.current_turn == 1:
				pos = None
				if self.human_mode:
					pos = input('Place a 1 onto an empty square: ')
				else:
					pos = self.requestInput(False)
				self.fill_space(1, pos)
				if self.filled_spaces >= self.start_searching_for_winner:
					self.search_for_winner()
				if self.found_winner == 0: self.current_turn = 2
			elif self.current_turn == 2:
				pos = self.requestInput(True)
				self.fill_space(-1, pos)
				if self.filled_spaces >= self.start_searching_for_winner:
					self.search_for_winner()
				if self.found_winner == 0: self.current_turn = 1
			if self.human_mode:
				print(self)

		if self.human_mode:
			if self.filled_spaces == self.num_blocks:
				print("It's a draw. There is no winner.")
			elif self.found_winner == 1:
				print("Game Over! Player 1 wins!")
			elif self.found_winner == 2:
				print("Game Over! Player 2 wins!")
		else:
			self.game_num += 1
			if self.filled_spaces == self.size ** self.ndim:
				if self.show_computer_progress: print("Game #{0: <6} It's a draw".format(self.game_num));
				self.draw_count += 1
				if self.training_mode:
					self.controller.game_result = 'draw'
			elif self.found_winner == 1:
				if self.show_computer_progress: print("Game #{0: <6} Computer 1 wins.".format(self.game_num));
				self.p1_win_count += 1
				if self.training_mode:
					self.controller.game_result = 'win'
			elif self.found_winner == 2:
				if self.show_computer_progress: print("Game #{0: <6} Computer 2 wins.".format(self.game_num));
				self.p2_win_count += 1
				if self.training_mode:
					self.controller.game_result = 'loss'

		if self.human_mode:
			should_restart = raw_input('Restart? y\\n: ')
			if should_restart == 'y' or should_restart == 'Y':
				self.restart()
			else:
				print("Thanks for playing!") 

	def restart(self):
		if self.current_turn == 1 : self.current_turn = 2
		else: self.current_turn = 1
		self.__init__(self.size, self.ndim, self.human_mode, self.controller, self.show_computer_progress)
		self.main()

class TicTacToeBot:

	def __init__(self, size=3, ndim=2, show_computer_progress=False, num_hidden_layers=1):
		self.size = size
		self.ndim = ndim
		self.num_blocks = size ** ndim
		self.show_computer_progress = show_computer_progress
		self.game_result = None
		self.num_hidden_layers = num_hidden_layers
		self.num_thetas = num_hidden_layers + 1
		self.thetas = ()
		self.trained = False

	def single_play(self, thetas):
		self.thetas = thetas
		self.game = GameBoard(self.size, self.ndim, False, self, False, True)
		self.game.current_turn = np.random.randint(2) + 1
		self.game.main()
		if self.game_result == 'win':
			return 1
		elif self.game_result == 'loss':
			return -1
		else:
			return 1

	def train(self, num_subjects, num_tribulations, num_generations, mutation_parameter, show_progress, start_randomly):

		if start_randomly:
			self.thetas = theta_generator(self.num_blocks, self.num_hidden_layers - 1, 1)

		last_performance = 0
		subjects = []
		

		for generation in range(num_generations):
			subjects = [None] * num_subjects
			for i in range(num_subjects):
				delta = theta_generator(self.num_blocks, self.num_hidden_layers - 1, mutation_parameter)
				subjects[i] = tuple(self.thetas[i] + delta[i] for i in range(self.num_thetas))

			results = [0] * num_subjects
			for i in range(num_subjects):
				for test in range(num_tribulations):
					results[i] += self.single_play(subjects[i])

			best_performance = max(results)
			if best_performance < last_performance:
				# do nothing. just go back to the last best and mutate it
				if show_progress:
					print colored("Generation {0:<5} Worse   {1}".format(generation, best_performance), 'red')  	
				continue
				
			if show_progress:
				if best_performance > last_performance:
					print colored("Generation {0:<5} Better  {1}".format(generation, best_performance), 'green')
				else:
					print colored("Generation {0:<5} Same    {1}".format(generation, best_performance), 'yellow')

			self.thetas = subjects[results.index(best_performance)]
			last_performance = best_performance

		self.trained = True
		print('Done training!')

	def load(self, *thetas):
		self.thetas = tuple(np.loadtxt(t) for t in thetas)
		self.num_thetas = len(thetas)
		self.num_hidden_layers = self.num_thetas - 1
		self.trained = True

	def play_self(self, num_iters):
		if not self.trained:
			print "Bot has not yet been trained. Both sides are playing randomly."

		self.game = GameBoard(self.size, self.ndim, False, self, self.show_computer_progress)
		for i in range(num_iters):
			self.game.restart()

		print("\n{0: <20} {1} matches.".format("Computer 1 won", self.game.p1_win_count))
		print("{0: <20} {1} matches.".format("Computer 2 won", self.game.p2_win_count))
		print("{0: <20} {1} matches.".format("It was a draw for", self.game.draw_count))
		print("---------------------------------")
		print("{0: <20} {1} matches".format("Total", self.game.game_num))

	def play_human(self):
		if not self.trained:
			print "Bot has not yet been trained. It is playing randomly."
		self.game = GameBoard(self.size, self.ndim, True, self, self.show_computer_progress)
		self.game.current_turn = np.random.randint(2) + 1
		self.game.restart()

	def provide_educated_position(self, current_turn):
		if self.trained:
			return neural_network_move(current_turn, self.thetas, self.game.board)
		else:
			return self.provide_random_position()

	def provide_random_position(self):
		choice = np.random.choice(self.game.plausible_moves)
		return np.unravel_index(choice, self.game.board.shape)

def theta_generator(size, extra_layers, mult=1):
	theta_start = (mult * (np.random.rand(hidden_layer_mult * size, size + 1) - 0.5),)
	theta_mids = tuple(mult * (np.random.rand(hidden_layer_mult * size, hidden_layer_mult * size + 1) - 0.5)
				for i in range(extra_layers))
	theta_end = (mult * (np.random.rand(size, hidden_layer_mult * size + 1) - 0.5),)
	return theta_start + theta_mids + theta_end

def neural_network_move(turn, thetas, board):
	a = board.flatten()
	if turn == 2:
		a *= -1
	vals = a
	for t in thetas:
		vals = np.dot(t, np.insert(vals, 0, 1))

	indices = np.nonzero(vals * (a == 0))[0]
	best_guess = indices[np.argmax(vals[indices])]
	return np.unravel_index(best_guess, board.shape)

# test = TicTacToeBot(num_hidden_layers=1)
# #test.load('theta111_improved.txt', 'theta222_improved.txt')
# test.train(20, 30, 100, 0.05, True, True)
# test.play_self(1000)
