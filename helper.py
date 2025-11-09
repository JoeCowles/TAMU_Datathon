import matplotlib.pyplot as plt

try:
    from IPython import display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

plt.ion()

def plot(scores, mean_scores):
    if IN_NOTEBOOK:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    if not IN_NOTEBOOK:
        plt.show(block=False)
        plt.pause(0.1)
