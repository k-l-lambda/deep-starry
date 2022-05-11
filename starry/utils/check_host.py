
import sys
import os
import subprocess



COMMANDS = os.getenv('CHECK_HOST_CMD')


def check (cmd):
	try:
		return subprocess.check_output(cmd).decode('utf-8')
	except:
		return str(sys.exc_info()[1])


def check_host ():
	result = ''

	commands = COMMANDS.split(';')
	for cmd in commands:
		result += f'> {cmd}\n'
		result += check(cmd.split(' ')) + '\n'

	return result
