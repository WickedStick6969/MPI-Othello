#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "comms.h" 

const int LENBUFSIZE=3;
const int MSGBUFSIZE=100;

int comms_get_colour(int* my_colour);

static int socket_desc;

/**
 * Creates socket, connects to remote server, and calls comms_get_colour 
 */
int comms_init_network(int* my_colour, unsigned long ip, int port) {
	struct sockaddr_in server;

	/* Create socket */
	socket_desc = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_desc == -1) {
		#ifdef DEBUG
		printf("Comms error: Could not create socket\n"); 
		#endif
		return FAILURE;
	}

	server.sin_addr.s_addr = ip;
	server.sin_family = AF_INET;
	server.sin_port = htons(port);

	/* Connect to remote server */
	if (connect(socket_desc, (struct sockaddr *)&server, sizeof(server)) < 0){
		#ifdef DEBUG
		printf("Comms error: Could not connect to server\n"); 
		#endif
		return FAILURE;
	}

	return comms_get_colour(my_colour);
}

/**
 * Creates socket, connects to remote server, and calls comms_get_colour 
 */
int comms_get_colour(int* my_colour) {
	char tempColour[2]; tempColour[1] = 0;
	if(recv(socket_desc, tempColour , 1, 0) < 0){
		#ifdef DEBUG
		printf("Comms error: Could not receive colour\n");
		#endif
		return FAILURE;
	}
	*my_colour = atoi(tempColour);
	return SUCCESS;
}

/**
 * Receives message from server, which includes a cmd 
 * and, if cmd == play_move, also the opponent's move 
 */
int comms_get_cmd(char cmd[], char move[]) {
	int result = SUCCESS;
	int msg_len;

	char* tmp_cmd;
	char* tmp_move;

	char* len_buf = malloc(LENBUFSIZE*sizeof(char));
	char* msg_buf = malloc(MSGBUFSIZE*sizeof(char)); 

	memset(len_buf, 0, LENBUFSIZE);
	memset(msg_buf, 0, MSGBUFSIZE);

	if (recv(socket_desc, len_buf , 2, 0) < 0){
		result = FAILURE;
	} else {

		msg_len = atoi(len_buf);
	
		if (recv(socket_desc, msg_buf, msg_len, 0) < 0){
			result = FAILURE; 
		} else {

			msg_buf[msg_len] = '\0';

			//strok is a dangerous command: 
			//- modifies argument
			//- returns a ptr into argument 
			//- maintains state, which is not thread-state  
			tmp_cmd = strtok(msg_buf, " ");  //get ptr to cmd in msg_buf; 
			tmp_move = strtok(NULL, " ");	//get ptr to move in msg_buf

			strncpy(cmd, tmp_cmd, CMDBUFSIZE); 
			if (tmp_move != NULL) strncpy(move, tmp_move, MOVEBUFSIZE);

			result = SUCCESS;
		}
	}

	free(len_buf);
	free(msg_buf);

	return result;
}

/**
 * Sends a message to the server, which includes my_move 
 * and, if cmd == play_move, also the opponent's move 
 */
int comms_send_move(char my_move[]) {

	if (send(socket_desc, my_move, strlen(my_move) , 0) < 0) {
		return FAILURE;
	}

	return SUCCESS;
}
