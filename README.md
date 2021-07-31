# UGV Code Repository

All source code for the control of the RoBros UGV for the UCRG internal competition challenges.

## Getting Started
We will be setting up a seperate catkin workspace and source code repository and linking them using symbolic links. This will allow us to build our project inside the workspace, while not interfering with the source code in our repo. For more on symbolic links refer to [this](https://www.makeuseof.com/tag/what-is-a-symbolic-link-what-are-its-uses-makeuseof-explains/). 

### Prerequisites
Make sure you have the following things installed:
- Running Linux (I'm running Ubuntu 16.04)
- ROS Kinetic

### Step 1
CD into the directory where you want to keep all your UCRG files. The final file structure will look something like:
- UCRG/
  - uav_repository/
    - src/
    - README.md
    - .gitignore
  - uav_workspace/
    - src/
    - devel/
    - build/
  - ugv_repository/
    - src/
    - README.md
    - .gitignore
  - ugv_workspace/
    - src/
    - devel/
    - build/
In your UCRG directory run:
```
git clone git@github.com:RoBros-1/ugv_repository.git
```
**or**
```
git clone https://github.com/RoBros-1/ugv_repository.git
```
### Step 2
Time to make our catkin workspace.
Run:
```
mkdir ugv_workspace/src -p
cd ugv_workspace
catkin_make
ls
```

### Step 3
Notice the **src/** directory in the workspace? The next step is to delete this directory and symbolically link the **src/** directory in your repo to the workspace. To do this, run:
```
rm src/ -r
ln -s ~/PATH/TO/YOUR/REPO/src/ .
ls
```

If you followed the file structure above, you can just copy thi following:
```
rm src/ -r
ln -s ../ugv_repository/src/ .
ls
```

Notice how the **src/** directory still shows up after `ls`, it might look different to other directories. This is because it is a symbolic link.

### Step 4
The final step is to build your workspace with the new source files. Simply run `catkin_make` again:
```
catkin_make
```

## Workflow
Now that you are set up, you will be doing most of you work in the workspace. The changes you make here will be reflected in the repository. When you are ready to commit your changes, commit to the appropriate branch from your repository. Be mindful of what you commit in each branch, and try to keep changes relevant to their branch.
