with import <nixpkgs> {};

python35.withPackages (ps: with ps; [
  praw     # A package to scrape comments from reddit
  requests # A rest client to scrape posts
])
