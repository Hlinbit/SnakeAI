import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, meam_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(meam_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(meam_scores) - 1, meam_scores[-1], str(meam_scores[-1]))