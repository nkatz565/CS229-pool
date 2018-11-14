import pygame
from pprint import pprint

from . import collisions
from . import event
from . import gamestate
from . import graphics
from . import config

was_closed = False
while not was_closed:
	game = gamestate.GameState()
	#button_pressed = graphics.draw_main_menu(game)

	#if button_pressed == config.play_game_button: - don't want to wait for button to be pressed
	game.start_pool()
	events = event.events()
	while not (events["closed"] or game.is_game_over or events["quit_to_main_menu"]):
		events = event.events()
		collisions.resolve_all_collisions(game.balls, game.holes, game.table_sides)
		
		game.redraw_all()
		#game.redraw_all_no_gphx() #UNCOMMENT THIS TO DISABLE DRAWING WHILE THE BALLS ARE MOVING. IF YOU WANT TO SAVE THE MOST TIME, USE THIS
		if game.all_not_moving():
			game.redraw_all()
			#game.redraw_all_no_gphx() #YOU CAN UNCOMMENT THIS, BUT IT ONLY RUNS ONCE SO ITS NOT A BIG DEAL
			#pprint(vars((game.return_game_state().balls[0])))
			game.check_pool_rules()
			game.cue.make_visible(game.current_player)
			while not (
				(events["closed"] or events["quit_to_main_menu"]) or game.is_game_over) and game.all_not_moving():
				game.redraw_all()
				#game.redraw_all_no_gphx() #UNCOMMENT THIS TO DISABLE SEEING THE BOARD WHILE IT'S WAITING FOR INPUT
				events = event.events()
				if game.cue.is_clicked(events):
					game.cue.cue_is_active(game, events)
				elif game.can_move_white_ball and game.white_ball.is_clicked(events):
					game.white_ball.is_active(game, game.is_behind_line_break())
	was_closed = events["closed"]

	#if button_pressed == config.exit_button:
	#	was_closed = True

pygame.quit()
