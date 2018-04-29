---
layout: default_sidebar
title: Notes and Samples on Python Topics
description: posted by ghbcode on 2015/01/17
---
# Virtual Environments (venv)
<br>
If you haven't heard of venv or the like I highly recommend that you look into them. What venv allows you to do is to create separtate environments usually with different settings and installed software. For example, you can have Python 2.7 installed in your machine and use venv to use Python 3.x. In windows the inner workings of venv may be more complicated, however, in Linux the 'trick' is simply that your environment variables are updated to reflect the installations in a particular folder. On my machine I'm running python 3.x inside of /Users/myusername/venvp3 so that if you output my \$PYTHONPATH you get:<br>
<br>

``` python
    > /usr/local/lib/python2.7/site-packages
```

<br><br>
However, if you activated my venv in ~/venvp3 and then you output my \$PYTHONPATH you would get:<br><br>

``` python
    > /Users/username/venvp3/lib/python3.5/site-packages
```    

<br><br>
It is really as simple as that, if you want to create an environment:
- (optional) Install a new version of Python such as 3.5.0
- create the virtual environment, in this case it will create it in the folder 'xyz' below
``` python
    > virtualenv -p /path/to/new/python xyz
```
- update the xyz/bin/activate file so that $PYTHONPATH points to the xyz/lib/site-packages and other locations where you may have modules


And now you can carry on as follows whenver you want to use this environment:<br>
- you go to the folder that is your particular venv
  - cd ~/my_venv
- you activate your environment which simply sets your environment variables to something specific (and useful)
  - source bin/activate
- you do whatever it is that you need to do
  - for example I start my ipython notebook that is running python 3.5
- once you are done, you revert back to your usual settings 
  - deactivate [this sets your environment variables back to your 'normal' settings]
<br><br>

And so you can have many different virtual environments with different settings. For instructions on how to set up virtualenv(what I call venv) and virtualenvwrapper (also very useful) in linux visit the links below:
- [virtual environment](https://wiki.archlinux.org/index.php/Python/Virtual_environment)
- [virtualenvwrapper](http://virtualenvwrapper.readthedocs.io/en/latest/)

### TIP

I keep a file called 'requirements.txt' in the xyz folder and in it I list all of the packages that I have installed via pip. The reason I do this is that I can use that requirements file to install a brand new virtual environment, with all of the packages in the file via this single command:

``` python
    > pip install -r requirements.txt
```
