# Welcome interns!
This guide will give you an overview of how to get started working on the SAGE project.

## Part 1: What is SAGE?
The Sage project's goal is to design and build a new kind of national-scale reusable cyberinfrastructure to enable AI at the edge. To learn about the project, check out the GitHub page, https://github.com/sagecontinuum/sage , and the SAGE project's website, https://sagecontinuum.org/ . I encourage you to explore the SAGE GitHub repository to learn the layout of the project. Be sure to write down any questions you may have while exploring the SAGE GitHub repository. When finshed overlooking the repository, contact Raj (<rajesh@anl.gov>) regarding questions about the SAGE GitHub repository.

## Part 2: Agile Scrum Development Process
Upon overviewing the repository, the next step is to go over the Agile Scrum framework. Agile Scrum is a workflow that promotes rapid prototyping and efficiency. The SAGE project is following the Agile Scrum develoment process. Every team member is encouraged to watch a six-part video series explaining the Agile Scrum process which can be found in the team's [Agile Scrum Google Drive](https://drive.google.com/drive/folders/1Cj1qPAEVZm1Qkd6BrbIAssERmHlmGf0C). **Download** (to enable document links to work) the latest "Software_Development_Process" PDF and head to section "Agile Scrum Training". The document also contains helpful tutorials on the tools the team uses (i.e. JIRA & Miro). If you have any questions please reach out to Joe Swantek (<jswantek@anl.gov>).

## Part 3: Docker
For the SAGE project, the application [Docker](https://www.docker.com/) is being used extensively. Docker is a tool designed to make it easier to create, deploy, and run applications by using containers. Take some time to learn the fundamentals of Docker. Here's a link to a short and friendly tutorial: https://www.youtube.com/watch?v=7S73WERRqO4. Feel free to watch or use any resources to learn about Docker (ex. [https://docs.docker.com/](https://docs.docker.com/)). You will want to have [Docker Desktop](https://www.docker.com/get-started) installed to your computer.

## Part 4: ...
Now that you have finished steps 1-3, report into Raj (<rajesh@anl.gov>). New steps will be added as we figure out what's going on :)



## Argonne Server Access Guide

### Account Setup

1. Generate an SSH public key. This must be the form of authentication that you use to SSH into any Argonne server. There is a good guide to generating a key [here](https://kb.iu.edu/d/aews).
2. Make an MCS account at https://accounts.mcs.anl.gov/. Under "reason for request", write what you will be using the server to do and specifically mention that you are working with Raj and Pete for the Waggle project. Enter a security question and finish up your new account request. (Note: eventually you will need to add your SSH public key this account, but in the request stage, I do not believe this is possible. Come back to it later.)
3. While you are waiting for your MCS account to be activated, create an LCRC account. This is the account that you will use to access services like Bebop (if you are assigned to that task). You can create one here: https://accounts.lcrc.anl.gov.
   1. Request membership to the `waggle` project by clicking "Join Project" on the right and then searching for waggle. Also add yourself to the `lcrc` project by doing the same thing. You should automatically be added to `lcrc`, but you will need to contact Raj or Pete when you request access to waggle, since they will need to accept the request.
   2. Add your public SSH key to the LCRC account. This should be the same one as you add to MCS.
4. Add your public SSH key to MCS when your account has been activated.

### SSH Setup

5. Add the configuration file `config` to your home `.ssh` folder on the machine that you will be using to login from. Paste this into that file:

   ```
   Host lcrc
           ProxyCommand ssh -q mcs nc -q0 bebop.lcrc.anl.gov 22
           User USERNAME
   Host mcs
           hostname login.mcs.anl.gov
           User USERNAME
           port 22
   ```

   Replace USERNAME with your LCRC and MCS usernames (ideally should be the same). This defines two hosts, lcrc and mcs. MCS is the outward-facing host that you should be able to access through a simple SSH connection. LCRC/Bebop is an internal host, so that is why we need to use MCS as a proxy to get to it.

6. Place your RSA private key (the one that came with your public key) into the `~/.ssh/rsa_id` file. This is where SSH will automatically pull your key from when you login.

7. Run the command `ssh mcs` to test your ability to log in to MCS. IMPORTANT: If it asks you for a password in a prompt that looks like `Password:`, you have done something wrong (likely you did not setup an `id_rsa` file correctly). The server should only need your private key, not an additional password. If you try your MCS or Argonne password, it will reject it, and in my case, block me from very essential Argonne services.

8. Run the command `ssh lcrc` to test your ability to login to Bebop. This should tunnel your connection through MCS and allow you to access Bebop.

9. In the Bebop SSH terminal, see if you can access the folder `/lcrc/project/waggle`. If you cannot, Raj or Pete have not accepted your request to join the waggle project on LCRC.
