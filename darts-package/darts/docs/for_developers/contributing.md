# Contributing to DARTS

## Get a local repository

To start, you need git access to open-darts. If you have *Developer* or *Maintainer* role in open-darts repository,
then you're set already.

To easly edit files clone the repository:

```bash
git clone git@gitlab.com:open-darts/open-darts.git
```

Alternatively, there is ``Clone or download`` button to copy the link to your clipboard.

you should now have a local directory named ``open-darts`` containing a Git repository that you can work in.

## Installation (development version)

## Create an issue

[**Issues**](https://gitlab.com/open-darts/open-darts/-/issues) are found in the left hand panel at [the repository home page](https://gitlab.com/open-darts/open-darts). Please check that the bug you have found or the feature you want to request does not already have an issue dedicated to it. If it does, feel free to comment on the discussion. If not, please make a new issue.

If you have discovered an error in open-darts, please describe

* What you were trying to achieve
* What you did to achieve this
* What result you expected
* What happened instead

With that information, it should be possible for someone else to reproduce the problem or to check that a proposed fix has really solved the issue.

### Make changes

Changes should be made on a new branch, dedicated to the issue you are addressing. That way, multiple people can work at the same time, and multiple issues can be addressed simultaneously, without everyone getting confused. Start creating a new branch from the `development` branch:

```bash
  git checkout development
  git checkout -b issue-ABC
```

where, instead of ABC, you put the number of the issue you're addressing.

Commits should be as small as possible. Please do not stack many changes for several days and then commit a whole giant bunch of changes in a single commit, as that makes it impossible to figure out later what was changed when, and which change introduced that bug we are trying to find.

**Never copy-paste code from another program!**. It's fine to have external dependencies, but those should be kept separate. Copy-pasting code leads to complicated legal issues. So please, only contribute code that you wrote yourself.

### Create a merge request

Once you have made all changes needed to resolve the issue, you should create a merge request. At this stage, your changes are on a branch, either in the main repository open-darts, or in a fork of the such repository. A merge request is a request to the maintainers of open-darts to incorporate the changes in your branch into the main version of open-darts.

Before creating a merge request, make sure that you have committed and pushed all your changes, and that the tests pass. To do so, run locally the tests in *darts-models* as follows:

```bash
python run_test_suite2.py
```

You should see OK printing for all tests. If there is a FAIL, please investigate why your changes are affecting such tests.

Go to the GitLab homepage of your fork, if you have one, or the main open-darts repository. On the left hand panel go to section *Merge requests*. Click on the button *New merge request*. In source branch you should select your branch and target branch should be `open-darts/open-darts/development`. Then click on *Compare branches and continue*.

This will lead you to a page describing your merge request. Add a description of the changes you've made, as well as other fields if you know them. Then finish the process by clicking on *Create merge request* at the end of the form.

### Code review

Like the issues, merge requests on Gitlab are a kind of discussion forum, in which the proposed changes can be discussed. Also the CI/CD will be triggered checking the compatibility of your proposed changes. The maintainers may ask you to make some improvements before accepting the merge request. While the merge request is open, any additional commits pushed to your public branch will automatically show up there.

Once we're all satisfied with the change, the pull request will be accepted, and your code will become part of open-darts.
