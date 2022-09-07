#ifndef _COMMS_H
#define _COMMS_H

#define FAILURE -1
#define SUCCESS 0

#define MOVEBUFSIZE 6
#define CMDBUFSIZE 100

int comms_init(int* my_colour);
int comms_init_network(int* my_colour, unsigned long ip, int port);
int comms_get_cmd(char cmd[], char move[]);
int comms_send_move(char move[]);

#endif
