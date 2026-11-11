# CFP selection committee protocol — what was verified, and where

Every claim below was read from a page fetched on 2026-11-09. The extracted text of each page is saved in
`docs/sources/`. "Official" means collegefootballplayoff.com. Press reports are marked as such.

Source keys used below:

| Key | Page |
|---|---|
| PROTOCOL | https://collegefootballplayoff.com/sports/2016/10/24/selection-committee-protocol (the live protocol page; despite the 2016 URL it carries 2026 dates, roster and recusals) |
| SC101 | https://collegefootballplayoff.com/news/2026/11/2/cfp-sc-101 |
| AUG26 | https://collegefootballplayoff.com/news/2026/8/6/sc-aug-2627.aspx |
| FORMAT26 | https://collegefootballplayoff.com/news/2026/1/23/2627-format.aspx |
| BOWLS26 | https://collegefootballplayoff.com/news/2026/2/3/2627-2728-bowls.aspx |
| RANK1 | https://collegefootballplayoff.com/news/2026/11/3/rankings |
| CHAIR-1103 | https://irishsportsdaily.com/s/28473/college-football-playoff-ranking-week-one-transcript (press transcript of the chair's Nov 3, 2026 call) |
| ASAP | https://asaptext.com/asap_media/media/1070/1122/ (ASAP Sports archive of CFP media conferences, Nov 2020 onward) |
| ASAP-251125 | https://asaptext.com/asap_media/media/1070/1122/transcripts/162235.html (Nov 25, 2025 call) |
| KSL25 | https://www.ksl.com/article/51363313/cfp-committee-adding-weight-to-games-against-strong-opponents-limit-reward-for-beating-weak-foes (AP report quoting the Aug 20, 2025 committee statement) |
| SDS25 | https://www.saturdaydownsouth.com/news/college-football/cfp-selection-committee-makes-change-to-strength-of-schedule-metric-for-2025 |
| SDS26 | https://www.saturdaydownsouth.com/news/college-football/cfp-committee-to-begin-considering-strength-of-conferences-in-selection-process/ |

## 1. Flagged conflicts — resolved

| Question | Answer | Source |
|---|---|---|
| Fifth automatic bid: highest-ranked G6 **champion** or highest-ranked G6 **team**? | **Team.** "the highest-ranked team from the American, Conference USA, Mid-American Conference, Mountain West, Pac-12 Conference or Sun Belt Conference". "Beginning this year" — this is new for 2026; winning the conference is not required. | AUG26, PROTOCOL, SC101 |
| Do the top three seeds choose their quarterfinal site? | **Yes.** The No. 1 seed selects its quarterfinal bowl *and* the semifinal it feeds; No. 2 then No. 3 select from the remaining quarterfinals and both feed the other semifinal; No. 4 gets the remaining quarterfinal. | PROTOCOL ("Selection Sequence"), AUG26, SC101 |
| Is the rebuilt Pac-12 a Group of Six league in 2026? | **Yes.** It is listed with the American, CUSA, MAC, Mountain West and Sun Belt as the six leagues whose top-ranked team gets the fifth bid. | PROTOCOL, AUG26 |

## 2. 2026 format

- Twelve teams. Automatic: champions of the ACC, Big Ten, Big 12, SEC, plus the highest-ranked team from the six
  leagues above. Then "the next seven highest-ranked teams". (AUG26, PROTOCOL)
- Notre Dame "will be included in the Playoff if it is ranked among the top 12 teams in the final CFP rankings".
  This is a guarantee, so a top-12 Notre Dame cannot be the at-large team bumped by a low-ranked automatic
  qualifier. (AUG26, PROTOCOL) *How this interacts with bumping is not spelled out on the official pages; the
  code treats Notre Dame in the top 12 as protected and bumps the lowest-ranked non-protected at-large team.*
- No cap on teams per conference. (PROTOCOL "Participants")
- Seeding is straight by ranking. "The four highest-ranked teams will be seeded 1, 2, 3 and 4 and will receive
  byes". Automatic qualifiers ranked outside the top 12 "will be seeded at the bottom of the 12 in rank order".
  (PROTOCOL "Pairings")
- If no G6 team is in the top 25, the committee identifies the highest-ranked one itself. If any automatic
  qualifier is outside the top 25 the committee determines its relative position. (PROTOCOL "Ranking of Unranked
  Automatic Qualifiers") *The model handles this by ranking every FBS team, not just 25.*
- First round: 5 hosts 12, 6 hosts 11, 7 hosts 10, 8 hosts 9, on campus. Quarterfinals: 1 v 8/9, 2 v 7/10,
  3 v 6/11, 4 v 5/12. Semifinals: (1/8/9) v (4/5/12) and (2/7/10) v (3/6/11). No reseeding. (PROTOCOL)
- Dates: first round Dec 18–19, 2026 (SC101). Quarterfinals Wed Dec 30 at the Fiesta Bowl; Fri Jan 1, 2027 at the
  Cotton, Peach and Rose Bowls. Semifinals Jan 14 at the Orange Bowl and Jan 15 at the Sugar Bowl. (BOWLS26)
  Championship Mon Jan 25, 2027, Allegiant Stadium, Las Vegas. (FORMAT26)
- Games after Selection Day are not considered. (PROTOCOL)

## 3. 2026 ranking releases

Tue Nov 3 (7 pm ET), Tue Nov 10 (9 pm), Tue Nov 17 (9 pm), Tue Nov 24 (7:30 pm), Tue Dec 1 (7 pm), and
Selection Day Sun Dec 6 (noon–3 pm), all on ESPN. Six rankings. (AUG26, SC101)
As of 2026-11-09 only the Nov 3 ranking has been released. (https://collegefootballplayoff.com/rankings)

## 4. Committee, 2026

Thirteen members (AUG26, SC101): Hunter Yurachek (chair; AD, Arkansas), Troy Dannen (AD, Nebraska),
Mark Dantonio (former HC, Cincinnati and Michigan State), Doug Gillin (AD, Appalachian State), Mark Harlan
(AD, Utah), Ivan Maisel (former sportswriter), Gus Malzahn (former HC, Arkansas State, Auburn, UCF),
Chris Massaro (AD, Middle Tennessee), Randall McDaniel (former OL, Arizona State), Mike Riley (former HC,
Oregon State and Nebraska), Jeff Tedford (former HC, California and Fresno State), Wesley Walls (former TE,
Ole Miss), Carla Williams (AD, Virginia).

Recusals (AUG26; full unless marked partial):

| School | Member |
|---|---|
| Appalachian State | Doug Gillin; Hunter Yurachek |
| Arizona | Mark Harlan (partial) |
| Arizona State | Randall McDaniel (partial) |
| Arkansas | Hunter Yurachek |
| Auburn | Gus Malzahn (partial) |
| California | Jeff Tedford (partial) |
| Fresno State | Jeff Tedford (partial) |
| Michigan State | Mark Dantonio |
| Middle Tennessee State | Chris Massaro |
| Nebraska | Troy Dannen |
| Ole Miss | Wesley Walls (partial) |
| Oregon State | Mike Riley |
| SMU | Gus Malzahn (partial) |
| Stanford | Ivan Maisel |
| Utah | Mark Harlan |
| Virginia | Carla Williams |

A fully recused member cannot vote on the team and leaves the room; a partially recused member may stay but
does not vote. A recused member cannot list the team in a listing step or rank a pool that contains it.
(PROTOCOL, SC101) **The model does not use the roster or the recusals.** They are recorded here for reference
only. With 13 voters and a points-sum ballot a recusal is a small effect, and there is no public ballot data to
estimate it from. An earlier version of this note said a `recused_votes` feature existed; it was never built.

## 5. Stated criteria

Mission: "select the best teams, rank the teams for inclusion in the playoff, and then assign the teams to the
playoff bracket." To distinguish "otherwise comparable teams" the committee considers: strength of schedule;
head-to-head competition; comparative outcomes of common opponents (without incenting margin of victory); and
other relevant factors such as unavailability of key players and coaches that may have affected performance
or likely will affect postseason performance. "There will not be one single metric." (PROTOCOL)

**Conference championships are no longer in the listed principles.** The 2026 protocol page lists four
criteria; "championships won", which was in the four-team-era protocol, is not among them. In 2026 a
conference title matters through the automatic bid, not as a stated tiebreaker. The model keeps a
champion flag and lets the data say what it is worth in each era. (PROTOCOL, SC101)

Three information sources: video, data through a customised portal (SportSource Analytics), and conference
input through two "point persons" per conference. (SC101, PROTOCOL)

## 6. Voting procedure (current)

Seven rounds, each with a listing step and a ranking step, all by secret ballot. (PROTOCOL, SC101)

1. Each member lists 30 teams, unordered. Teams named by three or more members stay under consideration
   (two for a team with a recused member).
2. Listing step: each member lists the best six remaining teams. The six with the most votes form the pool.
3. Ranking step: each member ranks the six, 1 point for first, 2 for second, and so on. The four teams with
   the fewest points are locked in as the next four. The other two are held over.
4. Each member lists six more; the top four vote-getters join the two holdovers.
5. Repeat for four rounds (ranks 1–4, 5–8, 9–12, 13–16).
6. For the last three rounds three teams are locked per round (17–19, 20–22, 23–25).
7. After any round, three or more members can vote to reconsider a group of three or more teams.

Modelling consequence: teams are compared inside pools of six, and the pool boundaries sit at 4/5, 8/9, 12/13
and 16/17. The chair confirmed on Nov 3, 2026 that head-to-head is only applied inside a pool: Alabama–Kentucky
was applied; Florida (24) over Ole Miss (10), South Carolina (21) over Tennessee (13) and Tennessee over Alabama
"is not going to come into play ... because of where those teams were reviewed in the ranking process".
(CHAIR-1103) The pairwise model therefore interacts head-to-head and common-opponent features with how close
two teams' résumés are.

## 7. Recent changes to what the committee is told to value

**2025 (announced Aug 20, 2025).** Committee statement as quoted by the AP and Saturday Down South: "The current
schedule strength metric has been adjusted to apply greater weight to games against strong opponents. An
additional metric, record strength, has been added to the selection committee's analysis to go beyond a team's
schedule strength to assess how a team performed against that schedule. This metric rewards teams defeating
high-quality opponents while minimizing the penalty for losing to such a team. Conversely, these changes will
provide minimal reward for defeating a lower-quality opponent while imposing a greater penalty for losing to
such a team." (KSL25, SDS25) *I could not locate the original release on collegefootballplayoff.com (the
guessed URL 404s), so the wording is taken from two press outlets that quote it identically.*

Also 2025: the top four seeds stopped being reserved for conference champions; seeding is straight by rank.
(PROTOCOL describes the current rule; the change year is from the brief and press, not re-verified officially.)

Executive director Rich Clark, Nov 25, 2025: "in our data, with our strength of schedule, margin of victory is
taken into account up to 24 points ... only as it pertains to strength of schedule", and members "look at the
12 correlative statistics that really are statistically indicative of strong football teams". (ASAP-251125)
The model's capped-margin features use a 24-point cap for that reason.

**2026.** "New for 2026-27, there is a Strength of Conference Enhancement, which is an enhancement, not a new
metric. A similar measure has been available to the committee in the past; this season it's presented with
greater emphasis on differences in schedule strength within a single conference." Conference strength "is
informed by the quality of the competition a team faces, the results it produces against that competition, and
measures including Opponent Team Strength, Schedule Strength and Record Strength." (SC101)

Press detail, not on the official site: ESPN's Heather Dinich reported the committee will "determine the
strength of each conference based on the schedule strength of each team in its league" to "help separate teams
in the SAME LEAGUE with similar records", and that it "won't trump quality wins, schedule strength and record
strength and what the committee sees on film". (SDS26) A late-November introduction of an additional
conference-strength data point for cross-conference comparisons is reported only by secondary sites. **Not
verified.** The formula for none of these metrics is public.

## 8. What the chair said on Nov 3, 2026 (CHAIR-1103)

- "Wins against teams that the committee thinks highly of are rewarded." "We are also not over-penalizing
  teams or losses that the committee thinks highly of." "a loss to Alabama or Tennessee or Georgia is not on the
  same level as an unranked team".
- Opponents at .500 or better were the stated yardstick: Alabama "seven of their eight games against teams
  that have a .500 record or better"; Indiana 5-1 in such games; Notre Dame 1-1.
- Opponent quality as of game time, including QB health: Mississippi State "at 4-0 at full strength" is
  "always" a quality win for Alabama "regardless of what the computer may say".
- A bad loss offset by a road win: Georgia No. 2 with "that one loss at home against Auburn, but they've got a
  really good win on the road against Alabama".
- Pods: the top four "were in a pod together ... joined by two other teams".
- Eye test used as a separator: Utah over UCLA with near-identical résumés ("just a better team when we watch
  them play"); Notre Dame over Wake Forest.
- Texas Tech (15) held down by a loss at unranked Cincinnati; LSU (14) ahead with zero ranked wins because
  both losses were to top-10 teams.
- Power-conference-vs-power-conference games are "a datapoint", taken team by team, not by conference.
- Ohio State: evaluated "as a team with another quarterback in the lineup"; "not being penalized" for the
  injury itself.
- Boise State (25) was the highest-ranked G6 team.

## 9. Could not verify

- The original collegefootballplayoff.com text of the Aug 20, 2025 metric announcement (press quotes only).
- The reported late-November conference-strength data point (secondary press only).
- The formulas for schedule strength, record strength and the conference-strength number (private).
- Chair transcripts before Nov 2020: the ASAP archive starts with the 2020 season.
