Code to read metranet files on CSCS

to enable git push/pull:
1. copy your public key to your github repo: [help page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

2. run the following command (each time):
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/*your_private_key*
```
