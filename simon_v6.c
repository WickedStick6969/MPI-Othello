/* vim: :se ai :se sw=4 :se ts=4 :se sts :se et */

/*H**********************************************************************
 *
 *    This is a skeleton to guide development of Othello engines that can be used
 *    with the Ingenious Framework and a Tournament Engine.
 *
 *    The communication with the referee is handled by an implementaiton of comms.h,
 *    All communication is performed at rank 0.
 *
 *    Board co-ordinates for moves start at the top left corner of the board i.e.
 *    if your engine wishes to place a piece at the top left corner,
 *    the "gen_move_master" function must return "00".
 *
 *    The match is played by making alternating calls to each engine's
 *    "gen_move_master" and "apply_opp_move" functions.
 *    The progression of a match is as follows:
 *        1. Call gen_move_master for black player
 *        2. Call apply_opp_move for white player, providing the black player's move
 *        3. Call gen move for white player
 *        4. Call apply_opp_move for black player, providing the white player's move
 *        .
 *        .
 *        .
 *        N. A player makes the final move and "game_over" is called for both players
 *
 *    IMPORTANT NOTE:
 *        Write any (debugging) output you would like to see to a file.
 *        	- This can be done using file fp, and fprintf()
 *        	- Don't forget to flush the stream
 *        	- Write a method to make this easier
 *        In a multiprocessor version
 *        	- each process should write debug info to its own file
 *H***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
#include "comms.h"

const int EMPTY = 0;
const int BLACK = 1;
const int WHITE = 2;

const int OUTER = 3;
const int ALLDIRECTIONS[8] = {-11, -10, -9, -1, 1, 9, 10, 11};
const int BOARDSIZE = 100;
const int TRUE = 1;
const int FALSE = 0;
const int SHARE = 1;

const int LEGALMOVSBUFSIZE = 65;
const char piecenames[4] = {'.', 'b', 'w', '?'};

void run_master(int argc, char *argv[]);
int initialise_master(int argc, char *argv[], int *time_limit, int *my_colour, FILE **fp);
void gen_move_master(char *move, int my_colour, FILE *fp);
void apply_opp_move(char *move, int my_colour, FILE *fp);
void game_over();
void run_worker();
void initialise_board();
void free_board();
int serial(char *move, int my_colour, FILE *fp);
void legal_moves(int player, int *moves, FILE *fp);
int legalp(int move, int player, FILE *fp);
int validp(int move);
int would_flip(int move, int dir, int player, FILE *fp);
int opponent(int player, FILE *fp);
int find_bracket_piece(int square, int dir, int player, FILE *fp);
int random_strategy(int my_colour, FILE *fp);
void make_move(int move, int player, FILE *fp);
void make_flips(int move, int dir, int player, FILE *fp);
int get_loc(char *movestring);
void get_move_string(int loc, char *ms);
void print_board(FILE *fp);
char nameof(int piece);
int count(int player, int *board);
int minimax(int move, int colour, int *sent_board, FILE *fp);
int alpha_beta(int move_made, int alpha, int beta, int colour, int depth, int *sent_board, FILE *fp);
int evaluate(int player, int *board, FILE *fp);
int game_stage();


int size;
int running;
int my_colour;
int time_limit;
int *board;
int rank;
int *moves;
int *scores;
FILE *fp;

int weights[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 120, -20, 20, 5, 5, 20, -20, 120, 0,
					0, -20, -40, -5, -5, -5, -5, -40, -20, 0,
					0, 20, -5, 15, 3, 3, 15, -5, 20, 0,
					0, 5, -5, 3, 3, 3, 3, -5, 5, 0,
					0, 5, -5, 3, 3, 3, 3, -5, 5, 0,
					0, 20, -5, 15, 3, 3, 15, -5, 20, 0,
					0, -20, -40, -5, -5, -5, -5, -40, -20, 0,
					0, 120, -20, 20, 5, 5, 20, -20, 120, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

typedef struct score_result
{
	int result;
	int move;
} score_result;

int main(int argc, char *argv[])
{

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
	{
		initialise_board();
		run_master(argc, argv);
		game_over();
	}
	else
	{
		initialise_board();
		run_worker(rank);
		MPI_Finalize();
	}
	return 0;
}

void run_master(int argc, char *argv[])
{
	char cmd[CMDBUFSIZE];
	char my_move[MOVEBUFSIZE];
	char opponent_move[MOVEBUFSIZE];

	running = 0;
	fp = NULL;

	if (initialise_master(argc, argv, &time_limit, &my_colour, &fp) != FAILURE)
	{
		running = 1;
	}
	if (my_colour == EMPTY)
		my_colour = BLACK;
	// Broadcast my_colour
	MPI_Bcast(&my_colour, 1, MPI_INT, 0, MPI_COMM_WORLD);

	while (running == 1)
	{
		/* Receive next command from referee */
		if (comms_get_cmd(cmd, opponent_move) == FAILURE)
		{
			fprintf(fp, "Error getting cmd\n");
			fflush(fp);
			running = 0;
			break;
		}

		/* Received game_over message */
		if (strcmp(cmd, "game_over") == 0)
		{
			running = 0;
			MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);
			fprintf(fp, "Game over\n");
			fflush(fp);
			break;

			/* Received gen_move message */
		}
		else if (strcmp(cmd, "gen_move") == 0)
		{
			// Broadcast running
			MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);
			// Broadcast board
			MPI_Bcast(board, BOARDSIZE, MPI_INT, 0, MPI_COMM_WORLD);

			gen_move_master(my_move, my_colour, fp);

			if (comms_send_move(my_move) == FAILURE)
			{
				running = 0;
				fprintf(fp, "Move send failed\n");
				fflush(fp);
				break;
			}
			print_board(fp);

			/* Received opponent's move (play_move mesage) */
		}
		else if (strcmp(cmd, "play_move") == 0)
		{
			apply_opp_move(opponent_move, my_colour, fp);
			print_board(fp);

			/* Received unknown message */
		}
		else
		{
			fprintf(fp, "Received unknown command from referee\n");
		}
	}
	// Broadcast running
	MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int initialise_master(int argc, char *argv[], int *time_limit, int *my_colour, FILE **fp)
{
	int result = FAILURE;

	if (argc == 5)
	{
		unsigned long ip = inet_addr(argv[1]);
		int port = atoi(argv[2]);
		*time_limit = atoi(argv[3]);

		*fp = fopen(argv[4], "w");
		if (*fp != NULL)
		{
			fprintf(*fp, "Initialise communication and get player colour \n");
			if (comms_init_network(my_colour, ip, port) != FAILURE)
			{
				result = SUCCESS;
			}
			fflush(*fp);
		}
		else
		{
			fprintf(stderr, "File %s could not be opened", argv[4]);
		}
	}
	else
	{
		fprintf(*fp, "Arguments: <ip> <port> <time_limit> <filename> \n");
	}

	return result;
}

void initialise_board()
{
	int i;
	board = (int *)malloc(BOARDSIZE * sizeof(int));
	for (i = 0; i <= 9; i++)
		board[i] = OUTER;
	for (i = 10; i <= 89; i++)
	{
		if (i % 10 >= 1 && i % 10 <= 8)
			board[i] = EMPTY;
		else
			board[i] = OUTER;
	}
	for (i = 90; i <= 99; i++)
		board[i] = OUTER;
	board[44] = WHITE;
	board[45] = BLACK;
	board[54] = BLACK;
	board[55] = WHITE;
}

void free_board()
{
	free(board);
}

/* The copyboard function mallocs space for a board, then copies
   the values of a given board to the newly malloced board.
*/

int *copy_board(int *board)
{
	int *new_board;
	new_board = (int *)malloc(BOARDSIZE * sizeof(int));
	for (int i = 0; i < BOARDSIZE; i++)
	{
		new_board[i] = board[i];
	}
	return new_board;
}

/**
 *   Rank i (i != 0) executes this code
 *   ----------------------------------
 *   Called at the start of execution on all ranks except for rank 0.
 *   - run_worker should play minimax from its move(s)
 *   - results should be send to Rank 0 for final selection of a move
 */
void run_worker()
{
	running = 0;
	int terminated = FALSE;
	int move;
	score_result score_result;
	// Broadcast colour
	MPI_Bcast(&my_colour, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// Broadcast running
	MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// result = malloc(2 * sizeof(int));

	while (running == 1)
	{
		// Broadcast board
		MPI_Bcast(board, BOARDSIZE, MPI_INT, 0, MPI_COMM_WORLD);
		// Generate move
		MPI_Status status;
		while (!terminated)
		{
			MPI_Recv(&move, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &status);
			score_result.move = move;

			if (move != -1)
			{

				score_result.result = minimax(move, my_colour, board, fp);

				MPI_Send(&score_result, 1, MPI_2INT, 0, 105, MPI_COMM_WORLD);
			}
			else
			{

				break;
			}
		}

		// Broadcast running
		MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
}

/**
 *  Rank 0 executes this code:
 *  --------------------------
 *  Called when the next move should be generated
 *  - gen_move_master should play minimax from its move(s)
 *  - the ranks may communicate during execution
 *  - final results should be gathered at rank 0 for final selection of a move
 */
void gen_move_master(char *move, int my_colour, FILE *fp)
{
	MPI_Status status;
	int loc, run, i, moves_sent;
	int moves_rec;
	int *rec_Moves, *rec_Scores;
	int best_score, best_move;
	score_result score_result;
	int *moves = (int *)malloc(LEGALMOVSBUFSIZE * sizeof(int));
	memset(moves, 0, LEGALMOVSBUFSIZE);
	// Obtains the legal moves to be shared among the other processes
	legal_moves(my_colour, moves, fp);

	rec_Moves = (int *)malloc(moves[0] * sizeof(int));
	rec_Scores = (int *)malloc(moves[0] * sizeof(int));

	moves_sent = 0;
	moves_rec = 0;
	loc = 0;

	// Run the program if serial if one thread is specified otherwise run the program in parallel.

	if (size == 1)
	{
		loc = serial(move, my_colour, fp);
		best_move = loc;
	}
	else
	{

		// Loop through the available processes
		for (i = 1; i < size; i++)
		{
			// Checks if the moves_sent to the processes are all sent and if it is terminate the loop
			if (moves_sent == moves[0])
			{
				break;
			}
			/*
			This if else statement forces the player to play the corner move regardless of
			the evalualtion. Hence making the algorithm stronger.
			*/
			if (moves[i] == 11)
			{
				loc = moves[i];
			}
			else if (moves[i] == 18)
			{
				loc = moves[i];
			}
			else if (moves[i] == 81)
			{
				loc = moves[i];
			}
			else if (moves[i] == 88)
			{
				loc = moves[i];
			}
			/*
				Send the moves out inititially to the processes
			*/
			MPI_Send(&moves[i], 1, MPI_INT, i, 100, MPI_COMM_WORLD);
			moves_sent++;
		}
		/* Another termination if statement, used to check if the moves_amt is greater than zero, and if it is not
			then set the run flag to -1 and sends a termination signal to the run_worker while loop which if the moves_amt
			is less than 0 will terminate the loop and send a pass.
		*/
		if (moves[0] <= 0)
		{
			run = -1;
			for (i = 1; i < size; i++)
			{
				MPI_Send(&run, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
			}
			strncpy(move, "pass\n", MOVEBUFSIZE);
		}
		/*
		This while loop is used to give dynamic work load balancing. It is used for when a score is sent back that
		it will give a processes more work if it has completed its job.
		*/
		int finished = FALSE;
		int flag;
		while (!finished)
		{
			/*The Iprobe is used to 'probe' for any signals on the tag of 105 which is where the results of the minimax
			gets sent from the run_worker*/
			MPI_Iprobe(MPI_ANY_SOURCE, 105, MPI_COMM_WORLD, &flag, &status);

			/*The loop blocks until it receives result data from run_worker, basically process 0 is waiting for its
				workers (run_worker) to send it its results from computing the minimax.
			*/
			if (flag == 1)
			{
				/*The process zero receives the result from the minimax struct in the run_worker function from the relevant
					MPI_send in run_worker once the specific process has executed the minimax.
				*/
				MPI_Recv(&score_result, 1, MPI_2INT, MPI_ANY_SOURCE, 105, MPI_COMM_WORLD, &status);
				/*
				Stores the result of the move computed by the workers in the rec_Moves and
				stores the actual result in the rec_scores array.
				then increases the amount of moves received.
				*/
				rec_Moves[moves_rec] = score_result.move;
				rec_Scores[moves_rec] = score_result.result;
				moves_rec++;

				/*
				Checks if all the moves received are equal to all the moves that are available and if they are
				sets the run flag to -1 and sends the signal to terminate to the workers.
				*/
				if (moves_rec == moves[0])
				{
					run = -1;
					for (i = 1; i < size; i++)
					{
						MPI_Send(&run, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
					}
					break;
				}
				else if (moves_sent < moves[0])
				{
					// Fixed the broken termination
					moves_sent++;
					MPI_Send(&moves[moves_sent], 1, MPI_INT, status.MPI_SOURCE, 100, MPI_COMM_WORLD);
				}
			}
		}

		best_score = -2000;
		best_move = rec_Moves[0];
		/*
		Loops through the moves received from the workers and updates the best move and score based on which one is greater
		than each other.
		*/
		for (i = 0; i < moves_rec; i++)
		{
			if (rec_Scores[i] > best_score)
			{
				best_score = rec_Scores[i];
				best_move = rec_Moves[i];
			}
		}
		// Very important to freeing up memory.
		free(rec_Moves);
		free(rec_Scores);
	}
	// Tell process zero to play the best move received from the workers.
	loc = best_move;

	if (loc == -1)
	{
		strncpy(move, "pass\n", MOVEBUFSIZE);
	}
	else
	{
		/* apply move */
		get_move_string(loc, move);
		make_move(loc, my_colour, fp);
	}
	free(moves);
}
/**
 * @brief A serial function that runs the minimax algorithm when the threads specified are equal to one
 *
 * @param move
 * @param my_colour
 * @param fp
 * @return int
 */
int serial(char *move, int my_colour, FILE *fp)
{

	int result, best_score, best_move;
	int *new_board;
	int *moves = (int *)malloc(LEGALMOVSBUFSIZE * sizeof(int));
	memset(moves, 0, LEGALMOVSBUFSIZE);
	legal_moves(my_colour, moves, fp);

	best_move = -100;
	best_score = -2000;

	if (moves[0] == 0)
	{
		free(moves);
		return -1;
	}

	new_board = copy_board(board);
	for (int i = 1; i <= moves[0]; i++)
	{

		result = minimax(moves[i], my_colour, board, fp);
		board = copy_board(new_board);
		make_move(moves[i], my_colour, fp);

		if (result > best_score)
		{
			best_move = moves[i];
			best_score = result;
		}
	}

	free(moves);
	free(new_board);
	return best_move;
}

void apply_opp_move(char *move, int my_colour, FILE *fp)
{
	int loc;
	if (strcmp(move, "pass\n") == 0)
	{
		return;
	}
	loc = get_loc(move);
	make_move(loc, opponent(my_colour, fp), fp);
}

void game_over()
{
	free_board();
	MPI_Finalize();
}

void get_move_string(int loc, char *ms)
{
	int row, col, new_loc;
	new_loc = loc - (9 + 2 * (loc / 10));
	row = new_loc / 8;
	col = new_loc % 8;
	ms[0] = row + '0';
	ms[1] = col + '0';
	ms[2] = '\n';
	ms[3] = 0;
}

int get_loc(char *movestring)
{
	int row, col;
	/* movestring of form "xy", x = row and y = column */
	row = movestring[0] - '0';
	col = movestring[1] - '0';
	return (10 * (row + 1)) + col + 1;
}

void legal_moves(int player, int *moves, FILE *fp)
{
	int move, i;
	moves[0] = 0;
	i = 0;
	for (move = 11; move <= 88; move++)
		if (legalp(move, player, fp))
		{
			i++;
			moves[i] = move;
		}
	moves[0] = i;
}

int legalp(int move, int player, FILE *fp)
{
	int i;
	if (!validp(move))
		return 0;
	if (board[move] == EMPTY)
	{
		i = 0;
		while (i <= 7 && !would_flip(move, ALLDIRECTIONS[i], player, fp))
			i++;
		if (i == 8)
			return 0;
		else
			return 1;
	}
	else
		return 0;
}
/*
A function adapted from the blog: Blog: https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
Used to get the number of moves the opponent or player can play in the evaluation function.
*/
int num_valid_moves(int player, int *board, FILE *fp)
{
	int i, move;
	i = 0;
	for (move = 11; move <= 88; move++)
	{
		if (legalp(move, player, fp))
		{
			i++;
		}
	}
	return i;
}

int validp(int move)
{
	if ((move >= 11) && (move <= 88) && (move % 10 >= 1) && (move % 10 <= 8))
		return 1;
	else
		return 0;
}

int would_flip(int move, int dir, int player, FILE *fp)
{
	int c;
	c = move + dir;
	if (board[c] == opponent(player, fp))
		return find_bracket_piece(c + dir, dir, player, fp);
	else
		return 0;
}

int find_bracket_piece(int square, int dir, int player, FILE *fp)
{
	while (board[square] == opponent(player, fp))
		square = square + dir;
	if (board[square] == player)
		return square;
	else
		return 0;
}

int opponent(int player, FILE *fp)
{
	if (player == BLACK)
		return WHITE;
	if (player == WHITE)
		return BLACK;
	fprintf(fp, "illegal player\n");
	return EMPTY;
}

int random_strategy(int my_colour, FILE *fp)
{
	int r;
	int *moves = (int *)malloc(LEGALMOVSBUFSIZE * sizeof(int));
	memset(moves, 0, LEGALMOVSBUFSIZE);

	legal_moves(my_colour, moves, fp);
	if (moves[0] == 0)
	{
		return -1;
	}
	srand(time(NULL));
	r = moves[(rand() % moves[0]) + 1];
	free(moves);
	return (r);
}

void make_move(int move, int player, FILE *fp)
{
	int i;
	board[move] = player;
	for (i = 0; i <= 7; i++)
		make_flips(move, ALLDIRECTIONS[i], player, fp);
}

void make_flips(int move, int dir, int player, FILE *fp)
{
	int bracketer, c;
	bracketer = would_flip(move, dir, player, fp);
	if (bracketer)
	{
		c = move + dir;
		do
		{
			board[c] = player;
			c = c + dir;
		} while (c != bracketer);
	}
}

void print_board(FILE *fp)
{
	int row, col;
	fprintf(fp, "   1 2 3 4 5 6 7 8 [%c=%d %c=%d]\n",
			nameof(BLACK), evaluate(BLACK, board, fp), nameof(WHITE), evaluate(WHITE, board, fp));
	for (row = 1; row <= 8; row++)
	{
		fprintf(fp, "%d  ", row);
		for (col = 1; col <= 8; col++)
			fprintf(fp, "%c ", nameof(board[col + (10 * row)]));
		fprintf(fp, "\n");
	}
	fflush(fp);
}

char nameof(int piece)
{
	assert(0 <= piece && piece < 5);
	return (piecenames[piece]);
}

int count(int player, int *board)
{
	int i, cnt;
	cnt = 0;
	for (i = 1; i <= 88; i++)
		if (board[i] == player)
			cnt++;
	return cnt;
}

/**
 * @brief A function to iteratively run through the moves available and increase the depth based on the amount.
 * Used to increase the speed of the program.
 *
 * @param moves
 * @return int
 */
int dynamic_depth(int moves)
{
	if (moves >= 3 && moves < 8)
	{
		return 6;
	}

	else if (moves >= 8 && moves < 15)
	{
		return 5;
	}

	else if (moves >= 15)
	{
		return 4;
	}

	else if (size == 1)
	{
		return 0;
	}

	else
	{
		return 7;
	}

	return 2;
}
/**
 * @brief The minimax starter function, obtains the legal moves of the player and the depth from the dynamic_depth
 * function.
 * computes the result from the alpha beta function with -10000 and 10000 as the alpha and beta functions respectively.
 *
 * @param move
 * @param colour
 * @param sent_board
 * @param fp
 * @return int
 */
int minimax(int move, int colour, int *sent_board, FILE *fp)
{
	int result, depth;
	int *moves = (int *)malloc(LEGALMOVSBUFSIZE * sizeof(int));
	memset(moves, 0, LEGALMOVSBUFSIZE);
	legal_moves(my_colour, moves, fp);
	int moves_avail = moves[0];
	depth = dynamic_depth(moves_avail);
	// printf("Depth:%d\n",depth);
	free(moves);

	result = alpha_beta(move, -10000, 10000, colour, depth, sent_board, fp);

	return result;
}

int alpha_beta(int move_made, int alpha, int beta, int colour, int depth, int *sent_board, FILE *fp)
{
	int *new_board;
	int result;
	int *moves = (int *)malloc(LEGALMOVSBUFSIZE * sizeof(int));
	
	memset(moves, 0, LEGALMOVSBUFSIZE);
	legal_moves(my_colour, moves, fp);
	// Makes my move
	new_board = copy_board(sent_board);

	if (depth == 0)
	{
		return evaluate(my_colour, new_board, fp);
	}
	// Gets the opponent legal moves
	make_move(move_made, colour, fp);
	new_board = copy_board(sent_board);
	legal_moves(opponent(colour, fp), moves, fp);
	// Checking if there no moves
	if (moves[0] == 0)
	{
		free(moves);
		return evaluate(my_colour, new_board, fp);
	}
	// Loop through the opponents set of moves and run the alpha beta
	for (int i = 1; i <= moves[0]; i++)
	{

		result = alpha_beta(moves[i], alpha, beta, opponent(colour, fp), depth - 1, new_board, fp);
		board = copy_board(new_board);
		make_move(moves[i], colour, fp);

		if (colour == my_colour)
		{
			if (result > alpha)
			{
				alpha = result;
			}
		}
		else
		{
			if (result < beta)
			{
				beta = result;
			}
		}
		// Prune
		if (alpha >= beta)
		{
			break;
		}
	}

	free(moves);
	free(new_board);
	if (colour == my_colour)
	{

		return alpha;
	}
	else
	{
		return beta;
	}
}


/**
 * @brief The game_stage function was inspired by a tutorial session where the demi informed me that
 * I would need to account for the later stages of the game and apply increased weights to the evaluation
 * function so that the minimax would not just pick the higher weight but rather where it would win.
 *
 * @return int
 */
int game_stage()
{
	int i;
	int pcoins;
	int ocoins;
	int opp = opponent(my_colour, fp);
	int total, stage;

	pcoins = 0;
	ocoins = 0;
	stage = 0;

	for (i = 1; i <= 88; i++)
	{
		if (board[i] == my_colour)
		{
			pcoins++;
		}
		else if (board[i] == opp)
		{
			ocoins++;
		}
	}

	total = pcoins + ocoins;

	if (total <= 20)
	{
		stage = 1;
	}
	else if (total > 20 && total <= 40)
	{
		stage = 2;
	}
	else if (total > 40)
	{
		stage = 3;
	}

	return stage;
}

/*
This evaluation function is inspired by mr peter sieg and a blog for a really good evaluation function that takes into
account several other heurisitcs such as coin parity, mobility and stability. This combined with the game_state function
gives me an optimal to play the othello game:
References are:
Blog: https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
Mr peter sieg github: https://github.com/petersieg/c
*/
int evaluate(int player, int *board, FILE *fp)
{
	int i;
	int pcnt, pcoins, pmoves;
	int opp, ocnt, ocoins, omoves;
	int position, parity, mobility, final;
	// opponent characteristics:
	opp = opponent(player, fp);
	ocnt = 0;
	ocoins = 0;
	omoves = num_valid_moves(player, board, fp);
	// player characteristics:
	pcnt = 0;
	pcoins = 0;
	pmoves = num_valid_moves(player, board, fp);

	for (i = 1; i <= 88; i++)
	{
		if (board[i] == player)
		{
			pcnt = pcnt + weights[i];
			pcoins++;
		}
		else if (board[i] == opp)
		{
			ocnt = ocnt + weights[i];
			ocoins++;
		}
	}
	position = (pcnt - ocnt);

	// Parity:
	parity = 100 * (pcoins - ocoins) / (pcoins + ocoins);
	// Mobility:
	if ((pmoves + omoves) != 0)
	{
		mobility = 100 * (pmoves - omoves) / (pmoves + omoves);
	}
	else
	{
		mobility = 0;
	}
	/*Programs runs with just the above*/
	// Corners Captured
	ocoins = pcoins = 0;
	if (board[11] == player)
		pcoins++;
	else if (board[11] == opp)
		ocoins++;

	if (board[18] == player)
		pcoins++;
	else if (board[18] == opp)
		ocoins++;

	if (board[81] == player)
		pcoins++;
	else if (board[81] == opp)
		ocoins++;

	if (board[88] == player)
		pcoins++;
	else if (board[88] == opp)
		ocoins++;

	int corner_occ = 25 * (pcoins - ocoins);

	// Corner Closeness:
	ocoins = pcoins = 0;
	if (board[11] == '.')
	{
		if (board[12] == player)
			pcoins++;
		else if (board[12] == opp)
			ocoins++;
		if (board[22] == player)
			pcoins++;
		else if (board[22] == opp)
			ocoins++;
		if (board[21] == player)
			pcoins++;
		else if (board[21] == opp)
			ocoins++;
	}
	if (board[18] == '.')
	{
		if (board[17] == player)
			pcoins++;
		else if (board[17] == opp)
			ocoins++;
		if (board[27] == player)
			pcoins++;
		else if (board[27] == opp)
			ocoins++;
		if (board[28] == player)
			pcoins++;
		else if (board[28] == opp)
			ocoins++;
	}

	if (board[81] == '.')
	{
		if (board[82] == player)
			pcoins++;
		else if (board[82] == opp)
			ocoins++;
		if (board[72] == player)
			pcoins++;
		else if (board[72] == opp)
			ocoins++;
		if (board[71] == player)
			pcoins++;
		else if (board[71] == opp)
			ocoins++;
	}

	if (board[88] == '.')
	{
		if (board[78] == player)
			pcoins++;
		else if (board[78] == opp)
			ocoins++;
		if (board[77] == player)
			pcoins++;
		else if (board[77] == opp)
			ocoins++;
		if (board[87] == player)
			pcoins++;
		else if (board[87] == opp)
			ocoins++;
	}

	int cc = -12.5 * (pcoins - ocoins);

	mobility = (3 - game_stage()) * mobility;

	final = position + (10 * parity) + (78.922 * mobility) + (801.724 * corner_occ) + (382.026 * cc);
	return final;
}
