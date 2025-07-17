import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    N = len(corpus)
    all_pages = corpus.keys()

    linked_pages = corpus[page]
    probabilities = dict()

    if linked_pages:
        for p in all_pages:
            prob = (1 - damping_factor) / N

            if p in linked_pages:
                prob += damping_factor / len(linked_pages)
            probabilities[p] = prob

    else:
        for p in all_pages:
            probabilities[p] = 1/N

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_page = random.choice(list(corpus.keys()))
    counts = {page: 0 for page in corpus}

    for _ in range(n):
        counts[current_page] += 1

        probs = transition_model(corpus, current_page, damping_factor)

        pages = list(probs.keys())
        weights = list(probs.values())
        current_page = random.choices(pages, weights=weights, k=1)[0]

    pagerank = {page: count / n for page, count in counts.items()}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}

    converged = False
    while not converged:
        new_ranks = {}
        for page in corpus:
            new_rank = (1 - damping_factor) / N
            total = 0
            for possible_page in corpus:
                if corpus[possible_page]:
                    if page in corpus[possible_page]:
                        total += pagerank[possible_page] / len(corpus[possible_page])
                else:
                    total += pagerank[possible_page] / N
            new_rank += damping_factor * total
            new_ranks[page] = new_rank

        converged = True
        for page in pagerank:
            if abs(new_ranks[page] - pagerank[page]) > 0.001:
                converged = False
                break

        pagerank = new_ranks

    return pagerank

if __name__ == "__main__":
    main()
