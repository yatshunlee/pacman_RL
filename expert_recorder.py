"""
The expert recorder.
"""
import argparse
import msvcrt
import gym
import json
import datetime

BINDINGS = {
    'w': 1,
    'a': 3,
    's': 4,
    'd': 2}


def get_options():
    parser = argparse.ArgumentParser(description='Records an expert..')
    parser.add_argument('data_directory', type=str,
                        help="The main datastore for this particular expert.")

    args = parser.parse_args()

    return args


def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory
    env = gym.make("ALE/MsPacman-ram-v5", render_mode='human')

    esc = False
    action = 0

    print("Welcome to the expert recorder")
    print("To record press:")
    print("w: turn up")
    print("a: turn left")
    print("s: turn down")
    print("d: turn right")
    print("Once you're finished press p to save the data.")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")

    while not esc:

        done = False
        obs = env.reset()
        obs = [int(o) for o in obs]

        ts = 0
        qlearn_pairs = {}
        while not done:
            # env.render(mode='human')
            # Handle the toggling of different application states

            # Take the current action if a key is pressed.
            # User input, but not displayed on the screen
            keys_pressed = msvcrt.getch().decode("utf-8")
            if keys_pressed == 'p':
                esc = True
                break

            if keys_pressed in BINDINGS:
                print(keys_pressed)
                action = BINDINGS[keys_pressed]
            else:
                action = action

            print(action)
            nxt_obs, reward, done, info = env.step(action)

            if esc:
                print("ENDING")
                break

            obs = [int(o) for o in obs]
            act = int(action)
            qlearn_pairs[ts] = (obs, act)


            ts +=1
            obs = nxt_obs

        if esc:
            break

    print("SAVING")
    filename = datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    with open(f'{ddir}/expert_{filename}.txt', 'w') as file:
        file.write(json.dumps(qlearn_pairs))


if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)