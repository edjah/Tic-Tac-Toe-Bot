# Tic-Tac-Toe-Bot


Info
-----
This is a simple implementation of 2-dimensional (n-dimensional coming soon) Tic-Tac-Toe. The only thing is that the computer has not been explicitly programmed to play Tic-Tac-Toe. It learns to play through trial and error (neural networks and evolutionary algorithsm)

Requirements
-------------
Tic-Tac-Toe-Bot runs on Python2. It is also dependent on a few modules
Use `pip2 install (module)` to install these if you don't already have them
* numpy
* termcolor

Instructions
-------------
0. Go to your command line
1. Clone this repository with `git clone https://github.com/edjah/Tic-Tac-Toe-Bot.git`
2. Go to the Tic-Toe-Toe-Bot directory with `cd Tic-Tac-Toe-Bot`
3. Start up python with `python2`
4. Enter `from tictac import TicTacToeBot`
5. Instantiate a new bot with `bot = TicToeToeBot()
6. Load the saved neural network parameters `theta1.txt` and `theta2.txt` with `bot.load('theta0.txt', 'theta1.txt')`
7. See how the trained bot (Computer 1) does when it plays against a random opponent with `bot.play_self(100)`
8. Play against the bot yourself with `bot.play_human()`
9. You will be prompted to place a 1 onto an empty square. Input a tuple of the form `(y,x)` where `(0,0)` would be the top-left corner of the board, and (2,2) would be the bottom-right corner of the board. For instance selecting `(0,2)` would get the following result in this situation.

[[0 0 0]
 [0 0 0]
 [0 0 0]]
Place a 1 onto an empty square: (0,2)
[[0 0 1]
 [0 0 0]
 [0 0 0]]
 
10. Play until you're satisfied. You're more than likely going to win the majority of the games since the bot has been trained to play against a random opponent. 

11. If you want to attempt to improve the neural network parameters of the bot, you can train it to improve on the current parameters with `bot.train(start_randomly=False)`. If you want to train the bot from scratch, you can use `bot.train(start_randomly=True)`. 

12. The training process consists of a set of candidate solutions which are extensively tested. The best of each generation is selected to be parent for the subsequent generation. The parent's children are individually randomly mutated with magnitude corresponding to the `mutation_parameter` variable in the train function's arguments. They are subsequently also tested. If one of the children outperforms the parent, it is selected to be the new parent, otherwise the old parent keeps its place. The process repeats, and it continues on for the selected number of generations, or until you press `CTRL + C`. Solution fitness is evaluated based on the number of wins and draws that a solution yielded. Each win adds 1 point to a solution's fitness score, a draw adds 0 points, and a loss subtracts 1 point. This has the effect of artificially selecting for solutions which produce more wins and draws than losses, but this can be modified in the `single_play` function in the `tictac.py` source code. The performance measure that is displayed when the `show_progress` flag is set to True, displays this score. By default, the maximum fitness score is equal to the number of tests, or `num_tribulations` that each candidate solution must go through, so candidates whose fitnesses approach this number are exceptional.
 `
13. Assess the performance of the bot after training by calling `bot.play_self(100)` once more.
14. If you're satisfied with your bots performance, you can save the neural network parameters with `bot.save(varname)` where `varname` is what you want your parameters to be called. This produces `n` files of the form `varname0.txt`, `varname1.txt`, `varname2.txt` ... `varname(n-1).txt` where `n` is the number of hidden layers you selected your network to have plus one.

15. That's pretty much it. Have fun. I'll be improving my selection mechanism in train soon, and adding n-dimensional support so you and the bot can play tic-tac-toe in 3 or more dimensions, so look forward to that.
